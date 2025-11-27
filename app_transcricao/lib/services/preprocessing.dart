import 'dart:typed_data';
import 'dart:math';
import 'package:fftea/fftea.dart';

class MelSpectrogram {
  final Float32List data; // tensor channel-first (nMels x paddedFrames)
  final int nFrames; // número real de frames (antes do padding)
  final int nMels; // número de bins Mel
  final int paddedFrames; // frames após padding para múltiplo de `padTo`

  MelSpectrogram(this.data, this.nFrames, this.nMels, this.paddedFrames);
}

/// Preprocessing compatível com NeMo Citrinet
///
/// Esta função recebe um array de samples (Float32List) e retorna um objeto
/// `MelSpectrogram` contendo: dados normalizados (channel-first), número de
/// frames originais, número de filtros Mel e número de frames com padding.
///
/// Parâmetros importantes:
/// - audio: áudio em ponto flutuante (espera-se faixa aproximada -1.0..1.0)
/// - sampleRate: taxa de amostragem (padrão 16kHz)
/// - nFft: tamanho do FFT (frequência bins = nFft/2)
/// - hopLength: salto entre frames (em samples)
/// - winLength: comprimento da janela (em samples)
/// - nMels: número de filtros Mel desejados
/// - fMin / fMax: faixa de frequência para o banco Mel
/// - eps: valor numérico pequeno para evitar log(0)
/// - padTo: padding para múltiplos (NeMo espera padding em blocos de 16)
/// - addDither: adiciona ruído muito pequeno para evitar problemas com sinais constantes
MelSpectrogram computeLogMelSpectrogram(
  Float32List audio, {
  int sampleRate = 16000,
  int nFft = 512,
  int hopLength = 160,
  int winLength = 400,
  int nMels = 80,
  double fMin = 0.0,
  double? fMax,
  double eps = 1e-10,
  int padTo = 16,
  bool addDither = false,
}) {
  // Se fMax não foi fornecido, usamos Nyquist (sr/2)
  fMax ??= sampleRate / 2.0;
  final fft = FFT(nFft);

  // Dither é um ruído muito fraco adicionado ao sinal para reduzir distorções causadas por quantização
  if (addDither) {
    final rnd = Random();
    for (int i = 0; i < audio.length; i++) {
      // adiciona ruído uniforme muito pequeno (ordem 1e-5)
      audio[i] += (rnd.nextDouble() * 2 - 1) * 1e-5;
    }
  }

  // --------------------
  // 1) Janela Hann
  // --------------------
  // Geração da janela de análise (Hann) usada em cada frame antes do FFT.
  final window = List<double>.generate(
    winLength,
    (i) => 0.5 - 0.5 * cos(2 * pi * i / (winLength - 1)),
  );

  // --------------------
  // 2) STFT (frames -> magnitude spectrogram)
  // --------------------
  // Para cada frame aplicamos janela, computamos FFT real e extraímos magnitude
  final List<List<double>> spec = [];
  for (int start = 0; start + winLength <= audio.length; start += hopLength) {
    // frame zerado com tamanho do nFft (padding zeros para FFT)
    final frame = List<double>.filled(nFft, 0.0);
    for (int i = 0; i < winLength; i++) {
      // aplica janela Hann ao frame
      frame[i] = audio[start + i] * window[i];
    }

    // realFft retorna pares (x,y) em cada bin — usamos magnitudes
    final spectrum = fft.realFft(frame);
    final mag = List<double>.generate(nFft ~/ 2, (i) {
      final c = spectrum[i];
      return sqrt(c.x * c.x + c.y * c.y);
    });
    spec.add(mag);
  }

  final nFreqs = nFft ~/ 2; // número de bins de frequência úteis
  final specLen = spec.length; // número de frames gerados

  // --------------------
  // 3) Banco de filtros Mel
  // --------------------
  // Cria uma matriz (nMels x nFreqs) que mapeia bins de FFT em bins Mel
  final filterbank = _melFilterbank(
    sr: sampleRate,
    nFft: nFft,
    nMels: nMels,
    fMin: fMin,
    fMax: fMax,
  );

  // --------------------
  // 4) Aplica filtros Mel e log
  // --------------------
  // Para cada frame multiplicamos pelo banco Mel (produto interno) e em seguida
  // aplicamos log(sum + eps) — o eps evita log(0).
  final melSpec = List.generate(specLen, (t) {
    final frame = spec[t];
    final melFrame = List<double>.filled(nMels, 0.0);
    for (int m = 0; m < nMels; m++) {
      double sum = 0.0;
      for (int f = 0; f < nFreqs; f++) {
        sum += filterbank[m][f] * frame[f];
      }
      melFrame[m] = log(sum + eps);
    }
    return melFrame;
  });

  // --------------------
  // 5) Padding para múltiplo de `padTo`
  // --------------------
  // NeMo geralmente espera uma dimensão temporal que seja múltiplo de 16
  final paddedFrames = ((specLen + padTo - 1) ~/ padTo) * padTo;
  final out = Float32List(nMels * paddedFrames);

  // --------------------
  // 6) Normalização por canal (média/variância) — estilo NeMo
  // --------------------
  // Normalizamos cada banda Mel individualmente (channel-first). Isso ajuda a
  // estabilizar a entrada para o modelo (remove offset e escala por banda).
  for (int m = 0; m < nMels; m++) {
    // coleta valores deste canal ao longo do tempo (antes do padding)
    final values = [for (int t = 0; t < specLen; t++) melSpec[t][m]];
    final mean = values.reduce((a, b) => a + b) / values.length;
    final variance = values.fold(0.0, (v, x) => v + pow(x - mean, 2)) / values.length;
    final std = sqrt(variance + 1e-10); // estabilidade numérica

    // Armazena no layout channel-first: index = m * paddedFrames + t
    for (int t = 0; t < specLen; t++) {
      out[m * paddedFrames + t] = ((melSpec[t][m] - mean) / std).toDouble();
    }
    // Padding temporal com zeros para completar até paddedFrames
    for (int t = specLen; t < paddedFrames; t++) {
      out[m * paddedFrames + t] = 0.0;
    }
  }

  // Retorna objeto com dados prontos para serem convertidos em tensor
  return MelSpectrogram(out, specLen, nMels, paddedFrames);
}

// --------------------
// Funções auxiliares: criação do banco de filtros Mel
// --------------------
List<List<double>> _melFilterbank({
  required int sr,
  required int nFft,
  required int nMels,
  required double fMin,
  required double fMax,
}) {
  final nFreqs = nFft ~/ 2;
  // Converte limites em escala Mel
  final melMin = _hzToMel(fMin);
  final melMax = _hzToMel(fMax);

  // Pontos mel igualmente espaçados (inclui 2 pontos extras para os limites)
  final melPoints = List<double>.generate(
    nMels + 2,
    (i) => melMin + (i * (melMax - melMin) / (nMels + 1)),
  );

  // Converte para Hertz e então para índices de bin do FFT
  final hzPoints = melPoints.map(_melToHz).toList();
  final bins = hzPoints.map((f) => ((nFft + 1) * f / sr).floor()).toList();

  // Inicializa matriz de zeros (nMels x nFreqs)
  final fb = List.generate(nMels, (_) => List<double>.filled(nFreqs, 0.0));

  // Para cada filtro Mel (triangular) construímos os slopes ascendentes/descendentes
  for (int m = 1; m <= nMels; m++) {
    final fLeft = bins[m - 1];
    final fCenter = bins[m];
    final fRight = bins[m + 1];
    
    // Slope ascendente: de fLeft até fCenter
    for (int f = fLeft; f < fCenter && f < nFreqs; f++) {
      if (fCenter > fLeft) {
        fb[m - 1][f] = (f - fLeft).toDouble() / (fCenter - fLeft);
      }
    }
    
    // Slope descendente: de fCenter até fRight
    for (int f = fCenter; f < fRight && f < nFreqs; f++) {
      if (fRight > fCenter) {
        fb[m - 1][f] = (fRight - f).toDouble() / (fRight - fCenter);
      }
    }
  }

  return fb;
}

// --------------------
// Conversões Hz <-> Mel
// --------------------
// Fórmulas padrão (HTK / Slaney style):
// Mel = 2595 * log10(1 + hz / 700)
// inverse: hz = 700 * (10^(mel/2595) - 1)

double _hzToMel(double hz) => 2595.0 * log(1.0 + hz / 700.0);
double _melToHz(double mel) => 700.0 * (exp(mel / 2595.0) - 1.0);

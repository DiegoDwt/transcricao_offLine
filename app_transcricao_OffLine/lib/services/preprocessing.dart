import 'dart:typed_data';
import 'dart:math';
import 'package:fftea/fftea.dart';

class MelSpectrogram {
  final Float32List data;
  final int nFrames;
  final int nMels;
  final int paddedFrames;

  MelSpectrogram(this.data, this.nFrames, this.nMels, this.paddedFrames);
}

/// Preprocessing compatível com NeMo Citrinet
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
  fMax ??= sampleRate / 2.0;
  final fft = FFT(nFft);

  // Dither opcional
  if (addDither) {
    final rnd = Random();
    for (int i = 0; i < audio.length; i++) {
      audio[i] += (rnd.nextDouble() * 2 - 1) * 1e-5;
    }
  }

  // Janela Hann 
  final window = List<double>.generate(
    winLength,
    (i) => 0.5 - 0.5 * cos(2 * pi * i / (winLength - 1)),
  );

  // STFT magnitude com sqrt
  final List<List<double>> spec = [];
  for (int start = 0; start + winLength <= audio.length; start += hopLength) {
    final frame = List<double>.filled(nFft, 0.0);
    for (int i = 0; i < winLength; i++) {
      frame[i] = audio[start + i] * window[i];
    }

    final spectrum = fft.realFft(frame);
    final mag = List<double>.generate(nFft ~/ 2, (i) {
      final c = spectrum[i];
      return sqrt(c.x * c.x + c.y * c.y);
    });
    spec.add(mag);
  }

  final nFreqs = nFft ~/ 2;
  final specLen = spec.length;

  // Banco de filtros Mel
  final filterbank = _melFilterbank(
    sr: sampleRate,
    nFft: nFft,
    nMels: nMels,
    fMin: fMin,
    fMax: fMax,
  );

  // Aplicar filtros Mel + log
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

  // Padding para múltiplo de 16
  final paddedFrames = ((specLen + padTo - 1) ~/ padTo) * padTo;
  final out = Float32List(nMels * paddedFrames);

  // Normalização por canal (NeMo-style: média/variância)
  for (int m = 0; m < nMels; m++) {
    final values = [for (int t = 0; t < specLen; t++) melSpec[t][m]];
    final mean = values.reduce((a, b) => a + b) / values.length;
    final variance = values.fold(0.0, (v, x) => v + pow(x - mean, 2)) / values.length;
    final std = sqrt(variance + 1e-10);

    // Normalizar e armazenar em layout channel-first
    for (int t = 0; t < specLen; t++) {
      out[m * paddedFrames + t] = ((melSpec[t][m] - mean) / std).toDouble();
    }
    // Padding com zeros
    for (int t = specLen; t < paddedFrames; t++) {
      out[m * paddedFrames + t] = 0.0;
    }
  }

  return MelSpectrogram(out, specLen, nMels, paddedFrames);
}

List<List<double>> _melFilterbank({
  required int sr,
  required int nFft,
  required int nMels,
  required double fMin,
  required double fMax,
}) {
  final nFreqs = nFft ~/ 2;
  final melMin = _hzToMel(fMin);
  final melMax = _hzToMel(fMax);

  final melPoints = List<double>.generate(
    nMels + 2,
    (i) => melMin + (i * (melMax - melMin) / (nMels + 1)),
  );

  final hzPoints = melPoints.map(_melToHz).toList();
  final bins = hzPoints.map((f) => ((nFft + 1) * f / sr).floor()).toList();

  final fb = List.generate(nMels, (_) => List<double>.filled(nFreqs, 0.0));

  for (int m = 1; m <= nMels; m++) {
    final fLeft = bins[m - 1];
    final fCenter = bins[m];
    final fRight = bins[m + 1];
    
    // Slope ascendente
    for (int f = fLeft; f < fCenter && f < nFreqs; f++) {
      if (fCenter > fLeft) {
        fb[m - 1][f] = (f - fLeft).toDouble() / (fCenter - fLeft);
      }
    }
    
    // Slope descendente
    for (int f = fCenter; f < fRight && f < nFreqs; f++) {
      if (fRight > fCenter) {
        fb[m - 1][f] = (fRight - f).toDouble() / (fRight - fCenter);
      }
    }
  }

  return fb;
}

double _hzToMel(double hz) => 2595.0 * log(1.0 + hz / 700.0);
double _melToHz(double mel) => 700.0 * (exp(mel / 2595.0) - 1.0);
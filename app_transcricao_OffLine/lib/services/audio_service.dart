import 'dart:io';
import 'dart:typed_data';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/foundation.dart';

class AudioService {
  final AudioRecorder _recorder = AudioRecorder();

  Future<String> startRecording() async {
    final hasPerm = await _recorder.hasPermission();
    if (!hasPerm) {
      throw Exception('Permissão do microfone não concedida');
    }

    final dir = await getApplicationDocumentsDirectory();
    final filePath = '${dir.path}/audio_${DateTime.now().millisecondsSinceEpoch}.wav';

    // Mono e 16kHz
    await _recorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        bitRate: 128000,
        sampleRate: 16000,
        numChannels: 1,  
      ),
      path: filePath,
    );

    return filePath;
  }

  Future<String?> stopRecording() async {
    return await _recorder.stop();
  }

  Future<Float32List> loadAudioAsFloat(String path) async {
    final file = File(path);
    final bytes = await file.readAsBytes();

    debugPrint("📊 Tamanho do arquivo WAV: ${bytes.length} bytes");

    // Lê o header WAV corretamente
    if (bytes.length < 44) {
      throw Exception("Arquivo WAV inválido (muito pequeno)");
    }

    // Verifica se é WAV válido
    final riff = String.fromCharCodes(bytes.sublist(0, 4));
    final wave = String.fromCharCodes(bytes.sublist(8, 12));
    
    if (riff != "RIFF" || wave != "WAVE") {
      throw Exception("Não é um arquivo WAV válido");
    }

    // Lê número de canais (byte 22-23)
    final numChannels = bytes[22] | (bytes[23] << 8);
    
    // Lê sample rate (bytes 24-27)
    final sampleRate = bytes[24] | 
                      (bytes[25] << 8) | 
                      (bytes[26] << 16) | 
                      (bytes[27] << 24);
    
    // Lê bits por sample (bytes 34-35)
    final bitsPerSample = bytes[34] | (bytes[35] << 8);
    
    debugPrint("📊 WAV Header:");
    debugPrint("   - Canais: $numChannels");
    debugPrint("   - Sample Rate: $sampleRate Hz");
    debugPrint("   - Bits per sample: $bitsPerSample");

    // Pular header (geralmente 44 bytes, mas pode variar)
    int dataStart = 44;
    
    // Procura pelo chunk "data"
    for (int i = 12; i < bytes.length - 8; i++) {
      if (bytes[i] == 0x64 && bytes[i+1] == 0x61 && 
          bytes[i+2] == 0x74 && bytes[i+3] == 0x61) {
        dataStart = i + 8;
        break;
      }
    }

    debugPrint("📊 Dados de áudio começam no byte: $dataStart");

    final pcmBytes = bytes.sublist(dataStart);
    final int16Data = Int16List.view(pcmBytes.buffer, pcmBytes.offsetInBytes);

    debugPrint("📊 Total de samples (int16): ${int16Data.length}");

    // Se for estéreo, converte para mono
    Float32List floatData;
    if (numChannels == 2) {
      debugPrint("⚠️ Áudio estéreo detectado, convertendo para mono...");
      final monoLength = int16Data.length ~/ 2;
      floatData = Float32List(monoLength);
      for (int i = 0; i < monoLength; i++) {
        // Média dos dois canais
        floatData[i] = (int16Data[i * 2] + int16Data[i * 2 + 1]) / 2.0 / 32768.0;
      }
      debugPrint("📊 Samples mono: ${floatData.length}");
    } else {
      // Já é mono
      floatData = Float32List(int16Data.length);
      for (int i = 0; i < int16Data.length; i++) {
        floatData[i] = int16Data[i] / 32768.0;
      }
      debugPrint("📊 Samples mono: ${floatData.length}");
    }

    // Verifica se não está todo zerado (silêncio)
    final nonZero = floatData.where((s) => s.abs() > 0.001).length;
    final percentActive = (nonZero / floatData.length * 100).toStringAsFixed(1);
    debugPrint("📊 Samples com sinal: $nonZero de ${floatData.length} ($percentActive%)");
    
    if (nonZero < floatData.length * 0.01) {
      debugPrint("⚠️ AVISO: Áudio parece estar quase todo em silêncio!");
    }

    // Mostra amplitude máxima
    final maxAmp = floatData.map((s) => s.abs()).reduce((a, b) => a > b ? a : b);
    debugPrint("📊 Amplitude máxima: ${(maxAmp * 100).toStringAsFixed(1)}%");
    
    if (maxAmp < 0.01) {
      debugPrint("⚠️ AVISO: Amplitude muito baixa! Fale mais alto ou ajuste o microfone.");
    }

    return floatData;
  }

  // Amplifica o áudio para melhorar a qualidade da transcrição
  Float32List amplifyAudio(Float32List audio, {double targetAmp = 0.3}) {
    // Encontra amplitude máxima
    final maxAmp = audio.map((s) => s.abs()).reduce((a, b) => a > b ? a : b);
    
    if (maxAmp < 0.001) {
      debugPrint("⚠️ Áudio muito baixo, não será amplificado");
      return audio;
    }
    
    // Calcula ganho para atingir targetAmp
    final gain = targetAmp / maxAmp;
    
    // Limita ganho máximo para evitar distorção excessiva
    final gainClamped = gain > 10.0 ? 10.0 : gain;
    
    debugPrint("🔊 Amplificando: ${(maxAmp * 100).toStringAsFixed(1)}% → ${(targetAmp * 100).toStringAsFixed(1)}% (ganho: ${gainClamped.toStringAsFixed(2)}x)");
    
    // Aplica ganho com clipping
    final amplified = Float32List(audio.length);
    for (int i = 0; i < audio.length; i++) {
      amplified[i] = (audio[i] * gainClamped).clamp(-1.0, 1.0);
    }
    
    return amplified;
  }
}
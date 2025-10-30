import 'dart:io';
import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:wav/wav.dart';

import 'services/preprocessing.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const TranscricaoApp());
}

class TranscricaoApp extends StatelessWidget {
  const TranscricaoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: TranscricaoPage(),
    );
  }
}

class TranscricaoPage extends StatefulWidget {
  const TranscricaoPage({super.key});

  @override
  State<TranscricaoPage> createState() => _TranscricaoPageState();
}

class _TranscricaoPageState extends State<TranscricaoPage> {
  OrtSession? _session;
  bool _isModelLoaded = false;
  List<String> _vocabulary = [];
  
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();
  String? _filePath;
  bool _isRecording = false;
  bool _isProcessing = false;
  String _result = "Aguardando gravação...";

  @override
  void initState() {
    super.initState();
    _loadVocabulary();
    _loadModel();
  }
  

  Future<void> _loadVocabulary() async {
    try {
      final vocabJson = await rootBundle.loadString('assets/models/labels.json');
      final List<dynamic> vocabList = json.decode(vocabJson);
      _vocabulary = vocabList.map((e) => e.toString()).toList();
    } catch (e) {
      if (kDebugMode) print("Erro ao carregar vocabulário: $e");
    }
  }

  Future<void> _loadModel() async {
    try {
      setState(() => _result = "Carregando modelo...");
      final raw = await rootBundle.load('assets/models/citrinet_encoder_decoder.onnx');
      final bytes = raw.buffer.asUint8List();
      OrtEnv.instance;
      final sessionOptions = OrtSessionOptions();
      _session = OrtSession.fromBuffer(bytes, sessionOptions);
      setState(() {
        _isModelLoaded = true;
        _result = "✅ Modelo carregado!\n\n🎤 Pressione o botão para gravar";
      });
    } catch (e) {
      setState(() => _result = "❌ Erro ao carregar modelo: $e");
    }
  }

  Future<void> _startRecording() async {
    if (await _recorder.hasPermission() == false) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Permissão de microfone negada'),
        ));
      }
      return;
    }
    
    final dir = await getTemporaryDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final filePath = '${dir.path}/audio_$timestamp.wav';
    
    await _recorder.start(
      RecordConfig(
        encoder: AudioEncoder.wav,
        bitRate: 128000,
        sampleRate: 16000,
        numChannels: 1,
      ),
      path: filePath,
    );
    
    setState(() {
      _isRecording = true;
      _filePath = filePath;
      _result = "🎤 GRAVANDO...\n\n⚠️ FALE BEM PRÓXIMO ao microfone\n⚠️ FALE ALTO e CLARO\n\nPressione novamente para parar";
    });
  }

  Future<void> _stopRecording() async {
    final path = await _recorder.stop();
    if (path == null) {
      setState(() => _isRecording = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Nenhuma gravação em andamento')),
        );
      }
      return;
    }
    
    setState(() {
      _isRecording = false;
      _filePath = path;
      _result = "✅ Gravação concluída!\n\nProcessando...";
    });
    
    await _processAudio(File(path));
  }

  Future<void> _processAudio(File file) async {
    if (_session == null) {
      setState(() => _result = "❌ Modelo não carregado.");
      return;
    }
    
    setState(() {
      _isProcessing = true;
      _result = "🔍 Processando áudio...";
    });
    
    try {
      final wavBytes = await file.readAsBytes();
      final wav = Wav.read(wavBytes);
      final Float32List audio = Float32List.fromList(wav.channels[0]);

      // Normalizar áudio
      double maxVal = audio.reduce((a, b) => a.abs() > b.abs() ? a : b).abs();
      final Float32List processedAudio = Float32List(audio.length);
      final targetMax = 0.95;
      final normFactor = maxVal > 0 ? targetMax / maxVal : 1.0;
      for (int i = 0; i < audio.length; i++) {
        processedAudio[i] = audio[i] * normFactor;
      }

      setState(() => _result = "🔍 Gerando espectrograma...");
      final MelSpectrogram mel = computeLogMelSpectrogram(processedAudio);

      setState(() => _result = "🔍 Executando modelo...");
      final featuresTensor = OrtValueTensor.createTensorWithDataList(
        mel.data,
        [1, mel.nMels, mel.paddedFrames],
      );
      final lengthTensor = OrtValueTensor.createTensorWithDataList(
        [mel.paddedFrames],
        [1],
      );
      final Map<String, OrtValue> inputs = {
        'features': featuresTensor,
        'features_len': lengthTensor,
      };

      final runOptions = OrtRunOptions();
      final outputs = await _session!.runAsync(runOptions, inputs)!.timeout(
        const Duration(seconds: 30),
      );

      if (outputs.isNotEmpty && outputs[0] != null) {
        final logits = outputs[0]!.value;
        final transcription = _decodeLogits(logits);
        final duration = (audio.length / wav.samplesPerSecond).toStringAsFixed(2);
        
        setState(() {
          _result = "✅ Transcrição concluída!\n\n"
              "━━━━━━━━━━━━━━━━━━━━━━\n"
              "📝 TEXTO TRANSCRITO:\n\n"
              "\"$transcription\"\n\n"
              "━━━━━━━━━━━━━━━━━━━━━━\n\n"
              "📊 Duração: ${duration}s\n\n"
              "🎤 Grave novamente ou reproduza o áudio";
        });
        
        for (final output in outputs) {
          output?.release();
        }
      } else {
        setState(() => _result = "❌ Nenhum output retornado.");
      }

      featuresTensor.release();
      lengthTensor.release();
      
    } catch (e) {
      setState(() => _result = "❌ Erro ao processar áudio:\n$e");
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _playRecording() async {
    if (_filePath == null || !File(_filePath!).existsSync()) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Nenhum áudio disponível')),
        );
      }
      return;
    }
    
    try {
      await _player.stop();
      await _player.play(DeviceFileSource(_filePath!));
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Erro ao reproduzir: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Transcrição Off-Line"),
        centerTitle: true, 
        backgroundColor: Colors.deepPurple,
        foregroundColor: Colors.white,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: _isModelLoaded
              ? Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      _isRecording ? Icons.mic : Icons.mic_none,
                      size: 80,
                      color: _isRecording ? Colors.red : Colors.grey,
                    ),
                    const SizedBox(height: 32),
                    Expanded(
                      child: SingleChildScrollView(
                        child: Text(
                          _result,
                          textAlign: TextAlign.center,
                          style: const TextStyle(fontSize: 16),
                        ),
                      ),
                    ),
                    const SizedBox(height: 32),
                    ElevatedButton.icon(
                      onPressed: (_isProcessing || _isRecording)
                          ? (_isRecording ? _stopRecording : null)
                          : _startRecording,
                      icon: Icon(_isRecording ? Icons.stop : Icons.fiber_manual_record),
                      label: Text(_isRecording ? 'Parar Gravação' : 'Iniciar Gravação'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _isRecording ? Colors.red : Colors.deepPurple,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                        disabledBackgroundColor: Colors.grey,
                      ),
                    ),
                    const SizedBox(height: 16),
                    ElevatedButton.icon(
                      onPressed: (_filePath != null && !_isProcessing && !_isRecording)
                          ? _playRecording
                          : null,
                      icon: const Icon(Icons.play_arrow),
                      label: const Text('Reproduzir Áudio'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                      ),
                    ),
                    if (_isProcessing) ...[
                      const SizedBox(height: 24),
                      const CircularProgressIndicator(),
                    ],
                  ],
                )
              : const Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text("Carregando modelo..."),
                  ],
                ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _recorder.dispose();
    _player.dispose();
    _session?.release();
    super.dispose();
  }

  String _decodeLogits(dynamic logits) {
    if (_vocabulary.isEmpty) {
      return "❌ Vocabulário não carregado";
    }
    
    try {
      List<List<double>> logitsMatrix;
      if (logits is List<List<List<double>>>) {
        logitsMatrix = logits[0].map((row) => List<double>.from(row)).toList();
      } else if (logits is List<List<double>>) {
        logitsMatrix = logits.map((row) => List<double>.from(row)).toList();
      } else {
        return "❌ Formato não suportado";
      }

      final result0 = _greedyDecode(logitsMatrix, 0);
      final result256 = _greedyDecode(logitsMatrix, _vocabulary.length - 1);
      
      final chosen = result256.length >= result0.length ? result256 : result0;

      StringBuffer text = StringBuffer();
      for (int i = 0; i < chosen.length; i++) {
        final idx = chosen[i];
        if (idx < _vocabulary.length) {
          final token = _vocabulary[idx];
          text.write(token.replaceAll('▁', ' '));
        }
      }

      final result = _postProcess(text.toString().trim());
      return result.isEmpty ? "(silêncio)" : result;
      
    } catch (e) {
      return "❌ Erro: $e";
    }
  }

  List<int> _greedyDecode(List<List<double>> matrix, int blank) {
    List<int> result = [];
    int prev = -1;
    
    for (var timestep in matrix) {
      int maxIdx = 0;
      double maxVal = timestep[0];
      for (int i = 1; i < timestep.length; i++) {
        if (timestep[i] > maxVal) {
          maxVal = timestep[i];
          maxIdx = i;
        }
      }
      
      if (maxIdx != blank && maxIdx != prev) {
        result.add(maxIdx);
      }
      prev = maxIdx;
    }
    
    return result;
  }

  String _postProcess(String text) {
    String result = text.trim();
    result = result.replaceAll(RegExp(r'\s+'), ' ');
    if (result.isNotEmpty) {
      result = result[0].toUpperCase() + result.substring(1);
    }
    result = result.replaceAllMapped(
      RegExp(r'\.\s+([a-z])'),
      (m) => '. ${m.group(1)!.toUpperCase()}',
    );
    if (result.isNotEmpty && !result.endsWith('.') && !result.endsWith('!') && !result.endsWith('?')) {
      result += '.';
    }
    return result;
  }
}
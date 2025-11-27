// main.dart

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle, TextInputFormatter, Clipboard, ClipboardData;
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:wav/wav.dart';
import 'services/preprocessing.dart';
import 'services/metrics.dart';

void main() async {
  // Garante que bindings do Flutter estão inicializados antes de qualquer operação
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const TranscricaoApp());
}

// Aplicação principal (Stateless) que encapsula localizações e o tema básico
class TranscricaoApp extends StatelessWidget {
  const TranscricaoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: const TranscricaoPage(),
      // Força locale pt-BR por padrão e inclui suporte a en-US
      locale: const Locale('pt', 'BR'),
      supportedLocales: const [
        Locale('pt', 'BR'),
        Locale('en', 'US'),
      ],
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
    );
  }
}

// Página principal da transcrição — Stateful pois mantém estado de gravação/modelo
class TranscricaoPage extends StatefulWidget {
  const TranscricaoPage({super.key});

  @override
  State<TranscricaoPage> createState() => _TranscricaoPageState();
}

class _TranscricaoPageState extends State<TranscricaoPage> {
  // Sessão ONNX para inferência (nullable até carregar o modelo)
  OrtSession? _session;
  bool _isModelLoaded = false; // flag indicando se o modelo está pronto
  List<String> _vocabulary = []; // labels / vocabulário para decodificação

  // Gravador e player de áudio
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();
  final TextEditingController _refController = TextEditingController(); // controlador pra referência de WER

  // controlador de rolagem adicionado para a transcrição — útil para textos longos
  final ScrollController _transcriptionScrollController = ScrollController();

  String? _filePath; // caminho do arquivo gravado
  bool _isRecording = false; // indica gravação em andamento
  bool _isProcessing = false; // indica processamento/inferência em andamento
  String _result = "Aguardando gravação..."; // texto principal com status / métricas

  // Texto transcrito (exibido no Card com botão copiar)
  String _transcription = '';

  // Modo de métricas (ativa coleta de métricas e cálculo de WER)
  bool _enableMetrics = false;
  Map<String, dynamic> _metrics = {};

  // Collector separado para métricas — inicializado em initState
  late final MetricsCollector _metricsCollector;

  @override
  void initState() {
    super.initState();
    _metricsCollector = MetricsCollector();
    _loadVocabulary(); // carrega labels do assets
    _loadModel(); // carrega o modelo ONNX do assets
  }

  @override
  void dispose() {
    // libera recursos e controllers ao destruir o widget
    _transcriptionScrollController.dispose();
    _recorder.dispose();
    _player.dispose();
    _session?.release();
    _refController.dispose();
    super.dispose();
  }

  // -------------------------
  // Funções auxiliares para copiar texto
  // -------------------------

  Future<void> _loadVocabulary() async {
    // Lê arquivo JSON de labels (assets/models/labels.json) e carrega _vocabulary
    try {
      final vocabJson = await rootBundle.loadString('assets/models/labels.json');
      final List<dynamic> vocabList = json.decode(vocabJson);
      _vocabulary = vocabList.map((e) => e.toString()).toList();
    } catch (e) {
      // Em modo debug, imprime erro de carregamento
      if (kDebugMode) print("Erro ao carregar vocabulário: $e");
    }
  }

  Future<void> _loadModel() async {
    // Carrega o modelo ONNX dos assets para uma sessão OrtSession
    try {
      setState(() => _result = "Carregando modelo...");
      final raw = await rootBundle.load('assets/models/citrinet_encoder_decoder.onnx');
      final bytes = raw.buffer.asUint8List();
      OrtEnv.instance; // inicializa ambiente ORT
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
    // Inicia gravação em WAV com configurações fixas (16kHz mono)
    if (await _recorder.hasPermission() == false) {
      // Se permissão negada, avisa o usuário
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Permissão de microfone negada'),
        ));
      }
      return;
    }

    // Gera caminho temporário para salvar arquivo WAV
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
      _transcription = ''; // limpa transcrição visual enquanto grava
      // reset scroll para o topo — útil quando o usuário grava várias vezes
      _transcriptionScrollController.jumpTo(0);
    });
  }

  Future<void> _stopRecording() async {
    // Para a gravação e inicia processamento caso exista um arquivo válido
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
    // Processa o WAV, extrai espectrograma, roda inferência e decodifica
    if (_session == null) {
      setState(() => _result = "❌ Modelo não carregado.");
      return;
    }

    setState(() {
      _isProcessing = true;
      _result = "🔍 Processando áudio...";
      if (!_enableMetrics) _metrics = {};
    });

    final stopwatchTotal = Stopwatch()..start();

    try {
      if (_enableMetrics) await _metricsCollector.snapshotPre();

      // Lê bytes do WAV e converte para Float32List com os samples do canal 0
      final wavBytes = await file.readAsBytes();
      final wav = Wav.read(wavBytes);
      final Float32List audio = Float32List.fromList(wav.channels[0]);

      // Normalizar áudio para evitar cortes / amplitude inconsistente
      double maxVal = audio.reduce((a, b) => a.abs() > b.abs() ? a : b).abs();
      final Float32List processedAudio = Float32List(audio.length);
      final targetMax = 0.95; // valor alvo após normalização
      final normFactor = maxVal > 0 ? targetMax / maxVal : 1.0;
      for (int i = 0; i < audio.length; i++) {
        processedAudio[i] = audio[i] * normFactor;
      }
      setState(() => _result = "🔍 Gerando espectrograma...");
      final stopwatchPreproc = Stopwatch()..start();
      // computeLogMelSpectrogram é uma função externa (services/preprocessing.dart)
      final MelSpectrogram mel = computeLogMelSpectrogram(processedAudio);
      stopwatchPreproc.stop();

      setState(() => _result = "🔍 Executando modelo...");
      final stopwatchModel = Stopwatch()..start();

      // Prepara tensores de entrada para a sessão ONNX
      final featuresTensor = OrtValueTensor.createTensorWithDataList(mel.data, [1, mel.nMels, mel.paddedFrames]);
      final lengthTensor = OrtValueTensor.createTensorWithDataList([mel.paddedFrames], [1]);
      final Map<String, OrtValue> inputs = {'features': featuresTensor, 'features_len': lengthTensor};
      final runOptions = OrtRunOptions();

      // Executa inferência (assíncrona) com timeout de 60s
      final outputs = await _session!.runAsync(runOptions, inputs)!.timeout(const Duration(seconds: 60));
      stopwatchModel.stop();

      final stopwatchDecode = Stopwatch()..start();
      String transcription = '';
      if (outputs.isNotEmpty && outputs[0] != null) {
        // outputs[0] esperado como logits — passa para decodificação
        final logits = outputs[0]!.value;
        transcription = _decodeLogits(logits);
        for (final output in outputs) output?.release();
      } else {
        transcription = "❌ Nenhum output retornado.";
      }
      stopwatchDecode.stop();

      stopwatchTotal.stop();

      // Duração em segundos aproximada (número de amostras / samplerate)
      final duration = (audio.length / wav.samplesPerSecond).toStringAsFixed(2);

      // snapshot post e obtenção de métricas (se ativado)
      Map<String, String> metricsResult = {};
      if (_enableMetrics) {
        metricsResult = await _metricsCollector.snapshotPost(stopwatchTotal.elapsedMilliseconds);
        _metrics = {
          'wall_ms': metricsResult['wall_ms'],
          'process_cpu_percent_approx': metricsResult['process_cpu_percent_approx'],
          'process_mem_mb_total': metricsResult['process_mem_mb_total'],
          'process_mem_mb_delta': metricsResult['process_mem_mb_delta'],
        };
      }

      // WER se modo ativado e referência preenchida — usa função estática do MetricsCollector
      double? wer;
      if (_enableMetrics) {
        final referenceText = _refController.text.trim();
        if (referenceText.isNotEmpty) {
          wer = MetricsCollector.computeWer(referenceText, transcription);
        }
      }

      // TRANSCRIÇÃO
      setState(() {
        _transcription = transcription; // exibida no Card com botão copiar

        // reseta a rolagem para o topo quando nova transcrição chega
        if (_transcriptionScrollController.hasClients) {
          _transcriptionScrollController.jumpTo(0);
        }

        // Monta apenas os textos de duração / métricas / tempos / WER (sem repetir a transcrição)
        if (_enableMetrics) {
          _result = "✅ Transcrição concluída!\n\n"
              "📊 Duração: ${duration}s\n"
              "⏱️ Tempo total (ms): ${stopwatchTotal.elapsedMilliseconds}\n"
              "⏱️ Pré-processamento (ms): ${stopwatchPreproc.elapsedMilliseconds}\n"
              "⏱️ Inferência (ms): ${stopwatchModel.elapsedMilliseconds}\n"
              "⏱️ Decodificação (ms): ${stopwatchDecode.elapsedMilliseconds}\n";

          if (_metrics.isNotEmpty) {
            // Adiciona métricas do dispositivo ao texto de resultado
            _result += "\n🔬 Métricas do dispositivo:\n";
            _result += "• CPU processo (aprox %): ${_metrics['process_cpu_percent_approx'] ?? 'N/A'}\n";
            _result += "• Memória total do processo (MB): ${_metrics['process_mem_mb_total'] ?? 'N/A'}\n";
            _result += "• Δ Memória (MB): ${_metrics['process_mem_mb_delta'] ?? 'N/A'}\n";
          }

          if (wer != null) _result += "\n📈 WER: ${(wer * 100).toStringAsFixed(2)}%\n";

          _result += "\n🎤 Grave novamente ou reproduza o áudio";
        } else {
          _result = "✅ Transcrição concluída!\n\n"
              "📊 Duração: ${duration}s\n"
              "⏱️ Tempo total: ${stopwatchTotal.elapsedMilliseconds}ms\n\n"
              "🎤 Grave novamente ou reproduza o áudio";
        }
      });

      // Libera tensores criados manualmente
      featuresTensor.release();
      lengthTensor.release();
    } catch (e) {
      setState(() => _result = "❌ Erro ao processar áudio:\n$e");
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _playRecording() async {
    // Reproduz o arquivo salvo em _filePath usando AudioPlayer
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
    // Use _transcription (separado) para exibir/copiar; remaining mostra _result (durações/métricas)
    final String transcriptionOnly = _transcription;
    final String remaining = _result;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Transcrição Off-Line"),
        centerTitle: true,
        backgroundColor: Colors.deepPurple,
        foregroundColor: Colors.white,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: _isModelLoaded
              ? Column(
                  children: [
                    const SizedBox(height: 8),
                    // Ícone grande que indica estado de gravação (ativo ou não)
                    Icon(
                      _isRecording ? Icons.mic : Icons.mic_none,
                      size: 80,
                      color: _isRecording ? Colors.red : Colors.grey,
                    ),
                    const SizedBox(height: 8),

                    // Switch para ativar/desativar coleta de métricas e WER
                    Row(
                      children: [
                        Expanded(
                          child: Center(
                            child: Text(
                              'Ativar métricas (CPU/Memória/WER):',
                              textAlign: TextAlign.center,
                            ),
                          ),
                        ),
                        Switch(
                          value: _enableMetrics,
                          onChanged: (v) {
                            setState(() {
                              _enableMetrics = v;
                              if (!v) {
                                // Limpa referência e métricas ao desativar
                                _refController.clear();
                                _metrics = {};
                                // remove possíveis blocos de métricas do texto de resultado
                                _result = _result.replaceAll(RegExp(r'\n🔬 Métricas do dispositivo:[\s\S]*'), '');
                                _result = _result.replaceAll(RegExp(r'\n📈 WER:.*'), '');
                              }
                            });
                          },
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),

                    // Campo para inserir transcrição de referência usado no cálculo de WER
                    if (_enableMetrics)
                      Localizations.override(
                        context: context,
                        locale: const Locale('pt', 'BR'),
                        child: TextField(
                          controller: _refController,
                          decoration: const InputDecoration(
                            labelText: 'Transcrição de referência (para WER)',
                            border: OutlineInputBorder(),
                            hintText: 'Digite a transcrição esperada',
                          ),
                          keyboardType: TextInputType.text,
                          textInputAction: TextInputAction.done,
                          maxLines: 2,
                          enableSuggestions: true,
                          autocorrect: true,
                          enableInteractiveSelection: true,
                          textCapitalization: TextCapitalization.sentences,
                          inputFormatters: <TextInputFormatter>[],
                        ),
                      ),

                    const SizedBox(height: 12),

                    // ---------- TRANSCRIÇÃO EM DESTAQUE COM BOTÃO LATERAL ----------
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Card(
                            elevation: 2,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                            child: Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  // Texto da transcrição (selecionável) — ocupa todo o espaço restante
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        const Text(
                                          'Transcrição',
                                          style: TextStyle(fontWeight: FontWeight.w600),
                                        ),
                                        const SizedBox(height: 6),

                                        // ConstrainedBox + Scrollbar + SingleChildScrollView para permitir rolagem da transcrição longa
                                        ConstrainedBox(
                                          constraints: BoxConstraints(
                                            // ajustável: altura máxima da área de transcrição dentro do card
                                            maxHeight: MediaQuery.of(context).size.height * 0.28,
                                          ),
                                          child: Scrollbar(
                                            controller: _transcriptionScrollController,
                                            thumbVisibility: true,
                                            child: SingleChildScrollView(
                                              controller: _transcriptionScrollController,
                                              padding: const EdgeInsets.only(right: 6),
                                              child: SelectableText(
                                                transcriptionOnly.isNotEmpty ? transcriptionOnly : '(vazio)',
                                                style: const TextStyle(fontSize: 15),
                                              ),
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),

                                  const SizedBox(width: 8),

                                  // Botão de copiar lateral — aparece apenas se houver transcrição
                                  Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      if (transcriptionOnly.isNotEmpty)
                                        IconButton(
                                          tooltip: 'Copiar transcrição',
                                          icon: const Icon(Icons.copy),
                                          onPressed: () {
                                            // Copia o texto da transcrição para a área de transferência
                                            Clipboard.setData(ClipboardData(text: transcriptionOnly));
                                            if (mounted) {
                                              ScaffoldMessenger.of(context).showSnackBar(
                                                const SnackBar(content: Text('Transcrição copiada para a área de transferência')),
                                              );
                                            }
                                          },
                                        )
                                      else
                                        const SizedBox(height: 48),
                                      const SizedBox(height: 2),
                                      const Text(
                                        'Copiar',
                                        style: TextStyle(fontSize: 12, color: Colors.black54),
                                      ),
                                    ],
                                  ),
                                ],
                              ),
                            ),
                          ),

                          const SizedBox(height: 12),

                          // ---------- REMANESCENTE (métricas / tempos / instruções) ----------
                          Expanded(
                            child: SingleChildScrollView(
                              child: Container(
                                width: double.infinity,
                                padding: const EdgeInsets.symmetric(horizontal: 8),
                                child: remaining.isNotEmpty
                                    ? SelectableText(
                                        remaining,
                                        textAlign: TextAlign.left,
                                        style: const TextStyle(fontSize: 14),
                                      )
                                    : const SizedBox.shrink(),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 12),

                    // Botão principal para iniciar/parar gravação
                    ElevatedButton.icon(
                      onPressed: (_isProcessing || _isRecording) ? (_isRecording ? _stopRecording : null) : _startRecording,
                      icon: Icon(_isRecording ? Icons.stop : Icons.fiber_manual_record),
                      label: Text(_isRecording ? 'Parar Gravação' : 'Iniciar Gravação'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _isRecording ? Colors.red : Colors.deepPurple,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
                      ),
                    ),
                    const SizedBox(height: 8),

                    // Botão para reproduzir áudio — habilitado apenas quando existe arquivo e não está processando
                    ElevatedButton.icon(
                      onPressed: (_filePath != null && !_isProcessing && !_isRecording) ? _playRecording : null,
                      icon: const Icon(Icons.play_arrow),
                      label: const Text('Reproduzir Áudio'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
                      ),
                    ),

                    if (_isProcessing) const Padding(padding: EdgeInsets.only(top: 12), child: CircularProgressIndicator()),
                    const SizedBox(height: 8),
                  ],
                )
              : const Column(mainAxisAlignment: MainAxisAlignment.center, children: [CircularProgressIndicator(), SizedBox(height: 16), Text("Carregando modelo...")]),
        ),
      ),
    );
  }

  String _decodeLogits(dynamic logits) {
    // Decodifica logits retornados pelo modelo em texto legível usando greedy decode
    if (_vocabulary.isEmpty) return "❌ Vocabulário não carregado";

    try {
      List<List<double>> logitsMatrix;
      if (logits is List<List<List<double>>>) {
        // Caso logits venha com batch extra (3D) — reduz para 2D
        logitsMatrix = logits[0].map((row) => List<double>.from(row)).toList();
      } else if (logits is List<List<double>>) {
        logitsMatrix = logits.map((row) => List<double>.from(row)).toList();
      } else {
        return "❌ Formato não suportado";
      }

      // Duas estratégias de decodificação: com blank=0 e blank=último token — escolhe a mais longa
      final result0 = _greedyDecode(logitsMatrix, 0);
      final result256 = _greedyDecode(logitsMatrix, _vocabulary.length - 1);
      final chosen = result256.length >= result0.length ? result256 : result0;

      StringBuffer text = StringBuffer();
      for (int i = 0; i < chosen.length; i++) {
        final idx = chosen[i];
        if (idx < _vocabulary.length) {
          final token = _vocabulary[idx];
          // Substitui caractere de subpalavra '▁' por espaço
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
    // Implementação simples de greedy decode com remoção de tokens repetidos e blank
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
      // Adiciona índice se não for blank e não repetir o anterior (CTC-like)
      if (maxIdx != blank && maxIdx != prev) result.add(maxIdx);
      prev = maxIdx;
    }
    return result;
  }

  String _postProcess(String text) {
    // Limpa espaços extras, capitaliza a primeira letra e garante pontuação final
    String result = text.trim();
    result = result.replaceAll(RegExp(r'\s+'), ' ');
    if (result.isNotEmpty) result = result[0].toUpperCase() + result.substring(1);
    // Capitaliza letra após ponto
    result = result.replaceAllMapped(RegExp(r'\.\s+([a-z])'), (m) => '. ${m.group(1)!.toUpperCase()}');
    // Adiciona ponto final caso não exista pontuação terminal
    if (result.isNotEmpty && !result.endsWith('.') && !result.endsWith('!') && !result.endsWith('?')) result += '.';
    return result;
  }
}

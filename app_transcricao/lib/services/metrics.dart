// metrics.dart

import 'dart:io';
import 'package:flutter/foundation.dart';

class MetricsCollector {
  // Valores lidos antes do processamento (em ticks/kB conforme /proc)
  int? _preUtimeTicks; // user-mode CPU ticks antes
  int? _preStimeTicks; // kernel-mode CPU ticks antes
  int? _preVmRssKb; // VmRSS (resident set size) em kB antes

  // --- Configurações ---
  // Número de núcleos assumido para cálculo aproximado de porcentagem de CPU.
  static const int _assumedCores = 8;
  // clk_tck aproximado (ticks por segundo) — padrão 100 em muitas distribuições Linux
  final int _clkTckApprox;

  MetricsCollector({int clkTckApprox = 100}) : _clkTckApprox = clkTckApprox;

  /// Leitura de /proc/self/stat (retorna mapa {utime, stime}) ou null
  /// - Em Android o arquivo /proc/self/stat contém vários campos sobre o
  ///   processo atual. Os campos utime (índice 13) e stime (índice 14) representam
  ///   os ticks de CPU gastos em user-mode e kernel-mode respectivamente.
  Future<Map<String,int>?> _readProcStat() async {
    try {
      // Só tentamos ler em ambientes Linux/Android, onde /proc existe
      if (!Platform.isAndroid && !Platform.isLinux) return null;
      final stat = await File('/proc/self/stat').readAsString();
      // O arquivo é uma linha com campos separados por espaços
      final parts = stat.split(RegExp(r'\s+'));
      // Índices conforme especificação do procfs: utime é campo 14 (index 13), stime é 15 (index 14)
      final utime = int.tryParse(parts[13]) ?? 0;
      final stime = int.tryParse(parts[14]) ?? 0;
      return {'utime': utime, 'stime': stime};
    } catch (e) {
      // Em caso de erro, loga somente em modo debug e retorna null
      if (kDebugMode) print('metrics._readProcStat erro: $e');
      return null;
    }
  }

  /// Tenta ler VmRSS do /proc/self/status (em kB) ou null
  /// - Em /proc/self/status a linha "VmRSS:" representa a memória residente em kB. 
  Future<int?> _readVmRssKb() async {
    try {
      if (!Platform.isAndroid && !Platform.isLinux) return null;
      final lines = await File('/proc/self/status').readAsLines();
      // Procura a primeira linha que começa com VmRSS:
      final line = lines.firstWhere((l) => l.startsWith('VmRSS:'), orElse: () => '');
      if (line.isEmpty) return null;
      // A linha tem formato: VmRSS:\t  12345 kB — dividimos por espaços e pegamos o número
      final parts = line.split(RegExp(r'\s+')).where((s) => s.isNotEmpty).toList();
      return int.tryParse(parts[1]);
    } catch (e) {
      if (kDebugMode) print('metrics._readVmRssKb erro: $e');
      return null;
    }
  }

  /// Chamar snapshotPre() imediatamente antes de iniciar uma operação
  /// que se deseja medir, para depois comparar no snapshotPost().
  Future<void> snapshotPre() async {
    try {
      final proc = await _readProcStat();
      _preUtimeTicks = proc?['utime'];
      _preStimeTicks = proc?['stime'];

      _preVmRssKb = await _readVmRssKb();

      if (kDebugMode) {
        print('metrics SNAPSHOT PRE: utime=$_preUtimeTicks stime=$_preStimeTicks vmrss=$_preVmRssKb');
      }
    } catch (e) {
      if (kDebugMode) print('metrics.snapshotPre erro: $e');
    }
  }

  /// Explicação do cálculo de CPU aproximado:
  /// - Leitura de utime+stime antes e depois fornece "ticks" de CPU gastos pelo processo.
  /// - Convertendo ticks em segundos: cpu_seconds = ticks_delta / clk_tck
  /// - Dividimos por wall-seconds e por número de núcleos para obter uma estimativa % do processador.
  Future<Map<String, String>> snapshotPost(int elapsedMs) async {
    try {
      final procPost = await _readProcStat();
      final postU = procPost?['utime'];
      final postS = procPost?['stime'];

      double? procCpuPercent;
      // Só calcula se houver snapshots válidos pré e pós
      if (_preUtimeTicks != null && postU != null && postS != null && _preStimeTicks != null) {
        final procTicksDelta = (postU + postS) - (_preUtimeTicks! + _preStimeTicks!);
        final seconds = elapsedMs / 1000.0;
        if (seconds > 0) {
          final cpuSeconds = procTicksDelta / _clkTckApprox;
          // Normaliza por tempo real decorrido e por número de núcleos (estimativa)
          procCpuPercent = (cpuSeconds / seconds) / _assumedCores * 100.0;
        }
      }
      // Lê o RSS pós-processamento (Resident Set Size em kB) para medir uso de memória atual
      final postVm = await _readVmRssKb();
      // Declara variáveis para memória total e delta (diferença) em MB, inicialmente nulas
      double? memMbTotal;
      double? memMbDelta;
      // Se o RSS pós for válido, converte de kB para MB para memória total
      if (postVm != null) memMbTotal = postVm / 1024.0; // kB -> MB
      // Calcula a diferença de memória (delta) se houver RSS pré e pós válidos
      if (_preVmRssKb != null && postVm != null) memMbDelta = (postVm - _preVmRssKb!) / 1024.0;
      
      // Cria um mapa com as métricas formatadas como strings para retorno
      final Map<String, String> result = {
        'wall_ms': elapsedMs.toString(),
        'process_cpu_percent_approx': procCpuPercent?.toStringAsFixed(2) ?? 'N/A',
        'process_mem_mb_total': memMbTotal?.toStringAsFixed(2) ?? 'N/A',
        'process_mem_mb_delta': memMbDelta?.toStringAsFixed(2) ?? 'N/A',
      };

      if (kDebugMode) print('metrics SNAPSHOT POST: $result');
      // Retorna o mapa de métricas calculadas
      return result;
    } catch (e) {
      if (kDebugMode) print('metrics.snapshotPost erro: $e');
      return {};
    }
  }

  /// Calcula WER entre referência e transcrição (strings).
  /// - Remove pontuação e normaliza para minúsculas antes de comparar.
  /// - Usa distância de Levenshtein em nível de palavra (edições / número de palavras de referência).
  /// - Retorna 0.0 quando a referência está vazia e hipótese também está (nenhum erro), ou 1.0 se
  ///   a referência estiver vazia e a hipótese não (tudo está errado).
  static double computeWer(String reference, String hypothesis) {
    // Normalização básica: trim e minúsculas
    String ref = reference.trim().toLowerCase();
    String hyp = hypothesis.trim().toLowerCase();

    // Remove caracteres não alfanuméricos (mantendo letras e números e espaços) — usa unicode flag
    ref = ref.replaceAll(RegExp(r"[^\p{L}\p{N}\s]", unicode: true), '');
    hyp = hyp.replaceAll(RegExp(r"[^\p{L}\p{N}\s]", unicode: true), '');

    // Split por espaços em palavras (removendo tokens vazios)
    final List<String> refWords = ref.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).toList();
    final List<String> hypWords = hyp.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).toList();

    final int n = refWords.length;
    final int m = hypWords.length;
    // Caso limite: referência vazia
    if (n == 0) return m == 0 ? 0.0 : 1.0;

    // DP table (n+1 x m+1) para distância de edição (Levenshtein) em nível de palavra
    final dp = List.generate(n + 1, (_) => List<int>.filled(m + 1, 0));
    for (int i = 0; i <= n; i++) {
      dp[i][0] = i; // deletions
    }
    for (int j = 0; j <= m; j++) {
      dp[0][j] = j; // insertions
    }

    for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= m; j++) {
        if (refWords[i - 1] == hypWords[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1]; // match
        } else {
          // substitution, insertion, deletion — pega o menor custo
          dp[i][j] = 1 + [dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]].reduce((a, b) => a < b ? a : b);
        }
      }
    }

    final edits = dp[n][m];
    // WER = número de edições / número de palavras da referência
    return edits / n;
  }
}

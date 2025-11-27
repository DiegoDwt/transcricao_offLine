Transcrição Off-Line de Áudio com Processamento On-Device (IA Embarcada)

É uma aplicação Flutter projetada para realizar transcrição de áudio totalmente off-line, utilizando modelos de Inteligência Artificial executados diretamente no dispositivo (on-device).
O projeto combina eficiência, privacidade e portabilidade, fornecendo uma solução robusta para captura, pré-processamento e reconhecimento de fala sem necessidade de conexão com a internet.

A aplicação faz uso de um pipeline de áudio otimizado que inclui normalização, geração de espectrogramas e inferência por modelos neurais compactos, garantindo desempenho adequado mesmo em dispositivos móveis com recursos limitados.

✨ Principais Características

Processamento 100% local: nenhum dado de áudio é enviado para servidores externos.
IA embarcada: modelos leves e otimizados para execução em CPU, utilizando  ONNX runtime.
Transcrição em tempo quase real: pipeline eficiente de captura e inferência.
Pré-processamento avançado de áudio:
normalização de amplitude (evitando clipping),
filtragem/remoção de ruído,
cálculo de STFT e banco Mel,
geração de espectrogramas prontos para o modelo.
Aplicativo multiplataforma Flutter: compatível com Android, iOS e desktops.
Privacidade garantida: todos os dados permanecem no dispositivo.

🎯 Objetivo

Oferecer uma alternativa confiável para transcrição de fala sem dependência da nuvem, ideal para casos de uso sensíveis à privacidade, ambientes sem conectividade ou aplicações profissionais que exigem controle total dos dados de áudio.

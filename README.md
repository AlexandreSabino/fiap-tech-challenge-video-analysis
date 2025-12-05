# Processamento de videos

## facial_recognition

Projeto que concentra os scritps para detecção facial no video e gera um report completo com anomalias e atividades encontradas.  
Artefatos: 
- facial_recognition/video_to_process.mp4 video original a ser processado.
- facial_recognition/processed.mp4 video processado com as marcações.
- facial_recognition/report.pdf report final com as métricas coletadas no video.

### Como rodar o projeto?

Instalação das libs
```bash
cd facial_recognition
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Rodar o arquivo principal:
```bash
python video_processor.py
```

## activity_detection

Projeto que concentra os scripts para detecção de atividades no video.
Artefatos: 
- activity_detection/video_to_process.mp4 video a ser processado.
- activity_detection/activities.json Resultado do processamento

### Como rodar o projeto?

Instalação das libs
```bash
cd activity_detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Rodar o arquivo principal:
```bash
python activity_detection.py
```

### ⚠️ Disclaimer:
O Projeto foi configurado e testado no Mac com processador M1!





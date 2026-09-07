# YoLo Edge API Maker

Este projeto trata-se de uma API para treinamento personalizado e implementação de sistemas de reconhecimento de objetos baseado no Yolo para sistemas embarcados de Edge AI baseado em Linux, em especial, para o Raspberry Pi. Em fundamento, este projeto trata-se de uma atividade do programa Intensivo Maker do PNAAT, realizado no Laboratório Maker Cariri, mas pretendo aperfeiçoá-lo para uso em projetos de código aberto.

**Autor: Pedro Henrique F.**

**Créditos: Programa PNAAT - Lab Maker Cariri**

---

## 1. Como preparar este repositório na sua máquina?

**Clone repositório:**

git clone https://github.com/pedrohfernandesbzr/yolo-edge-api


### Dentro da pasta ~/yolo-edge-api:

**Baixar dependências:**

pip install ultralytics flask opencv-python-headless pyyaml dvc --break-system-packages

**Recuperação de Dados (DVC):**
dvc pull

**Teste de Validação (Quality Gate):**
PYTHONIOENCODING=utf-8 python3 scripts/inspect_dataset.py --dataset dataset/exports/epi-v1/data.yaml --min-per-class 30 

---

## 2. Iniciando a Aplicação (Captura e Telemetria)

(campo a ser editado, pois script de interface ainda em andamento)
(mas é possível rodar os scripts do projeto diretamente)

### Para gravação de vídeo com inferência ao-vivo:

**Por camera IP:**
python3 ~/yolo-edge-api/stream/v3_optimized.py --fallback-url "http://[SEU_IP_AQUI]:8080/video" --infer-every [Pulo de frames] --infer-size 320 --output /tmp/stream_anotado.avi


### Para inferência ao-vivo em servidor web:

**Se estiver utilizando câmera por IP, use:**
python3 ~/yolo-edge-api/stream/mjpeg_server.py --fallback-url "http://[SEU_IP_AQUI]:8080/video" --infer-every [Número de pulo de frames] --infer-size 320


### Para captura de frames com objetico de teinamento do modelo:

**Captura via Câmera IP:**
python3 ~/yolo-edge-api/stream/capture_frames.py --fallback-url "http://[SEU_IP_AQUI]:8080/video" [NUMERO_CAPTURAS] --interval [INTERVALO_CAPTURAS_EM_SEGUNDOS]

**Captura via Câmera RPI Flet ou USB:**
python3 ~/yolo-edge-api/stream/capture_frames.py --total [NUMERO_CAPTURAS] --interval [INTERVADO_CAPTURAS_EM_SEGUNDOS]

**Para capturas manuais (precionando enter):**
Ao invés de por "--intervall 1.0", por exemplo (para capturas a cada 1 segundo), ponha:
"--manual"


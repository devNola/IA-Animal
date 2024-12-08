import cv2
import numpy as np

# Caminhos dos arquivos
ARQUIVO_CFG = "yolov3.cfg"
ARQUIVO_PESOS = "yolov3.weights"
ARQUIVO_CLASSES = "coco.names"

# Carregar as classes
with open(ARQUIVO_CLASSES, "r") as arquivo:
    CLASSES = [linha.strip() for linha in arquivo.readlines()]

# Gerar cores para cada classe
CORES = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# ID da classe "cachorro" no COCO dataset é 16
ID_CACHORRO = 16

def carregar_modelo_pretreinado():
    try:
        modelo = cv2.dnn.readNetFromDarknet(ARQUIVO_CFG, ARQUIVO_PESOS)
        modelo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        modelo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Ou DNN_TARGET_CUDA se tiver GPU
        return modelo
    except cv2.error as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

def detectar_movimento_e_cachorro(captura_de_video, modelo, largura_reduzida, altura_reduzida):
    cachorro_detectado = False
    while True:
        ret, quadro = captura_de_video.read()
        if not ret:
            print("fim do vídeo, obrigado por assistir.")
            break

        # Reduzir a resolução do quadro
        quadro_original = quadro.copy()
        quadro = cv2.resize(quadro, (largura_reduzida, altura_reduzida))

        # Janela 1: Detecção de movimento com YOLO
        altura, largura, _ = quadro.shape
        blob = cv2.dnn.blobFromImage(quadro, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        modelo.setInput(blob)
        camadas_saida = modelo.getUnconnectedOutLayersNames()

        detections = modelo.forward(camadas_saida)

        cachorro_area = None

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                classe_id = np.argmax(scores)
                confiança = scores[classe_id]
                if confiança > 0.5 and classe_id == ID_CACHORRO:  # Focando no cachorro
                    centro_x = int(obj[0] * largura)
                    centro_y = int(obj[1] * altura)
                    w = int(obj[2] * largura)
                    h = int(obj[3] * altura)

                    x = int(centro_x - w / 2)
                    y = int(centro_y - h / 2)

                    cachorro_area = (x, y, w, h)
                    cachorro_detectado = True

                    # Desenhar o retângulo e a classe no quadro
                    cv2.rectangle(quadro, (x, y), (x + w, y + h), CORES[classe_id], 2)
                    cv2.putText(quadro, "Cachorro", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, CORES[classe_id], 2)

        # Janela 2: Efeito criativo no fundo com foco no cachorro
        if cachorro_detectado and cachorro_area is not None:
            x, y, w, h = cachorro_area

            # Criar fundo desfocado
            fundo_desfocado = cv2.GaussianBlur(quadro_original, (21, 21), 0)

            # Isolar a área do cachorro no fundo desfocado
            # A área do cachorro não será desfocada, mantendo o cachorro nítido
            fundo_desfocado[y:y + h, x:x + w] = quadro_original[y:y + h, x:x + w]

            # Aplicar bordas brilhantes ao cachorro para destacá-lo
            cv2.rectangle(fundo_desfocado, (x, y), (x + w, y + h), (0, 255, 255), 4)  # Borda brilhante em amarelo
            cv2.putText(fundo_desfocado, "Cachorro", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Mostrar a detecção com fundo desfocado e cachorro destacado
            cv2.imshow("Fundo Desfocado - Cachorro Destaque", fundo_desfocado)

        # Janela 1: Mostra a detecção de movimento do cachorro normalmente
        if cachorro_detectado:
            cv2.imshow("Detecção de Movimento - Cachorro", quadro)

        # Tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    modelo = carregar_modelo_pretreinado()
    if modelo is None:
        print("Falha ao carregar o modelo YOLO.")
        return

    captura_de_video = cv2.VideoCapture('video-movimento.mp4')
    if not captura_de_video.isOpened():
        print("Erro ao abrir o arquivo de vídeo.")
        return

    # Reduzindo a resolução do vídeo para aumentar o desempenho
    largura_reduzida = 640
    altura_reduzida = 360

    # Detectar movimento e foco no cachorro
    detectar_movimento_e_cachorro(captura_de_video, modelo, largura_reduzida, altura_reduzida)

    captura_de_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import os
import cv2
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configuração do MediaPipe Tasks
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Arquivo modelo não encontrado em: {MODEL_PATH}")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.05 
)
detector = vision.HandLandmarker.create_from_options(options)

def extract_landmarks(image):
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    if not detection_result.hand_landmarks:
        return None

    hand = detection_result.hand_landmarks[0]
    wrist_x, wrist_y, wrist_z = hand[0].x, hand[0].y, hand[0].z
    
    landmarks = []
    for lm in hand:
        landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
    return landmarks

# 2. Caminho do dataset
DATASET_DIR = r"C:\projetos\codigoLibras\Libras-20260913T224959Z-1-001"

data = []
labels = []
total_arquivos = 0

# Extensões aceitas ( opções como .jfif, .tiff, .webp)
EXT_IMAGEM = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.jfif', '.tiff')
EXT_VIDEO = ('.mp4', '.avi', '.mov', '.mkv')

print(f"Processando dataset em: {DATASET_DIR} ...\n")

if os.path.exists(DATASET_DIR):
    for letra in os.listdir(DATASET_DIR):
        letra_dir = os.path.join(DATASET_DIR, letra)
        
        if os.path.isdir(letra_dir):
            letra_rotulo = letra.upper()
            
            for filename in os.listdir(letra_dir):
                file_path = os.path.join(letra_dir, filename)
                ext = filename.lower()

                # Processa IMAGENS
                if ext.endswith(EXT_IMAGEM):
                    total_arquivos += 1
                    img = cv2.imread(file_path)
                    
                    if img is None:
                        print(f" [Erro ao abrir arquivo]: {letra}/{filename}")
                        continue
                        
                    features = extract_landmarks(img)
                    if features:
                        data.append(features)
                        labels.append(letra_rotulo)
                        print(f" [Sucesso]: Hand detectada em {letra}/{filename}")
                    else:
                        print(f" [Mão NÃO detectada]: {letra}/{filename}")

                # Processa VÍDEOS
                elif ext.endswith(EXT_VIDEO):
                    total_arquivos += 1
                    cap = cv2.VideoCapture(file_path)
                    frame_count = 0
                    frames_com_mao = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_count % 5 == 0:
                            features = extract_landmarks(frame)
                            if features:
                                data.append(features)
                                labels.append(letra_rotulo)
                                frames_com_mao += 1
                        frame_count += 1
                    cap.release()
                    print(f" [Vídeo Processado]: {letra}/{filename} ({frames_com_mao} frames extraídos)")
                
                else:
                    print(f" [Extensão ignorada]: {letra}/{filename}")

    # 3. Salva no CSV e mostra o relatório final
    if data:
        df = pd.DataFrame(data)
        df['label'] = labels
        csv_path = os.path.join(SCRIPT_DIR, "dataset_libras.csv")
        df.to_csv(csv_path, index=False)
        
        print("\n")
        print(" CONTAGEM FINAL DE AMOSTRAS POR LETRA ")
        print(df['label'].value_counts().sort_index())
        print("\n")
        
        alfabeto_esperado = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        letras_encontradas = sorted(df['label'].unique())
        letras_faltantes = [l for l in alfabeto_esperado if l not in letras_encontradas]
        
        print(f"\nTotal de letras com amostras: {len(letras_encontradas)}")
        if letras_faltantes:
            print(f" Faltam amostras para as letras: {letras_faltantes}")
        else:
            print(" Todas as 26 letras possuem amostras salvas!")
    else:
        print("\nNenhuma mão foi detectada em nenhuma mídia.")
else:
    print(f"Erro: O diretório '{DATASET_DIR}' não foi encontrado.")
import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Carrega o modelo treinado e o detector do MediaPipe
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL_PATH = os.path.join(SCRIPT_DIR, 'modelo_libras.pkl')
MODEL_TASK_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

if not os.path.exists(MODEL_PKL_PATH):
    raise FileNotFoundError("Execute o treinar.py primeiro para gerar o modelo_libras.pkl!")

clf = joblib.load(MODEL_PKL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Inicializa a Webcam
cap = cv2.VideoCapture(0)

print("Iniciando webcam... Pressione 't' na janela do vídeo para sair.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inverte o frame horizontalmente (efeito espelho)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand = detection_result.hand_landmarks[0]
        wrist_x, wrist_y, wrist_z = hand[0].x, hand[0].y, hand[0].z
        
        # Extrai e normaliza os pontos da mão
        landmarks = []
        for lm in hand:
            landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
        
        # Faz a predição da letra
        prediction = clf.predict([landmarks])[0]
        
        # Exibe a letra na tela
        cv2.putText(frame, f"Letra: {prediction}", (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Reconhecedor de Libras em Tempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
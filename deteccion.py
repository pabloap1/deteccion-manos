import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Título de la app
st.title("Detección de Manos IA (Compatible con Python 3.13)")
st.write("Cargando modelo y preparando cámara...")

# 1. Configurar MediaPipe Tasks
@st.cache_resource
def load_hand_landmarker():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)

# Intentar cargar el detector
try:
    detector = load_hand_landmarker()
except Exception as e:
    st.error(f"Error cargando el modelo: {e}. Asegúrate de tener 'hand_landmarker.task' en la carpeta.")
    st.stop()

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Efecto espejo

        # Convertir BGR a RGB para MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        # Detección (modo Video requiere timestamp)
        timestamp = int(frame.pts * 1000 / frame.time_base.denominator)
        result = detector.detect_for_video(mp_image, timestamp)

        # Dibujar resultados si existen
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                # Dibujar puntos y líneas manualmente (reemplaza cvzone)
                h, w, _ = img.shape
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                
                # Opcional: Dibujar conexiones (simplificado)
                connections = mp.solutions.hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_lm = landmarks[start_idx]
                    end_lm = landmarks[end_idx]
                    start_point = (int(start_lm.x * w), int(start_lm.y * h))
                    end_point = (int(end_lm.x * w), int(end_lm.y * h))
                    cv2.line(img, start_point, end_point, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

# Configuración de WebRTC
webrtc_streamer(
    key="hand-detection-tasks",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=VideoProcessor().recv,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

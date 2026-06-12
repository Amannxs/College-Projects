import streamlit as st
import cv2
import numpy as np
import time
import torch
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import from app (functions & constants only - NOT face_mesh)
from app import compute_EAR, crop_single_eye, predict_eye, LEFT_EYE, RIGHT_EYE, device, model, transform, CLASS_NAMES

# Import MediaPipe locally to avoid import issues
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    MEDIAPIPE_OK = True
except Exception as e:
    st.error(f"❌ MediaPipe initialization failed: {e}")
    face_mesh = None
    MEDIAPIPE_OK = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="SafeDrive AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ SafeDrive AI: Professional Drowsiness Detection")

# --- SIDEBAR & STATE ---
st.sidebar.header("🔧 System Controls")
run_app = st.sidebar.toggle("System Power", value=False)
EAR_THRESHOLD = st.sidebar.slider("EAR Sensitivity", 0.15, 0.35, 0.23)
DROWSY_TIME = st.sidebar.slider("Alarm Delay (sec)", 1.0, 5.0, 2.0)

# Variables for tracking
if 'closed_start_time' not in st.session_state:
    st.session_state.closed_start_time = None
    st.session_state.alarm_on = False

# --- LAYOUT ---
col1, col2 = st.columns([3, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    st.subheader("Real-time Stats")
    status_ui = st.empty()
    ear_metric = st.empty()
    timer_metric = st.empty()
    st.divider()
    st.info("💡 Tip: Ensure good lighting on your face for better accuracy.")

# --- MAIN ENGINE ---
if not MEDIAPIPE_OK:
    st.error("❌ MediaPipe failed to initialize. Cannot proceed.")
elif run_app:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Make sure it's connected and has permission.")
    else:
        st.info("✅ Webcam connected. System running...")
        
        frame_count = 0
        while run_app and cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.error("❌ Webcam access lost!")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. MediaPipe Face Mesh Processing
            results = face_mesh.process(rgb_frame)
            
            is_drowsy = False
            avg_ear = 0.0

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # 2. EAR Logic
                left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
                right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # 3. Eye Cropping (for future CNN integration)
                left_crop, left_box = crop_single_eye(frame, landmarks, LEFT_EYE, w, h)
                right_crop, right_box = crop_single_eye(frame, landmarks, RIGHT_EYE, w, h)
                
                # Decision: EAR-based drowsiness
                if avg_ear < EAR_THRESHOLD:
                    is_drowsy = True
                
                # Draw Landmarks for visual feedback
                for idx in LEFT_EYE + RIGHT_EYE:
                    x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Draw bounding boxes around eyes
                for box, label in [(left_box, "L"), (right_box, "R")]:
                    if box:
                        x1, y1, x2, y2 = box
                        color = (0, 0, 255) if is_drowsy else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 4. Timer & Drowsiness Logic
            duration = 0.0
            if is_drowsy:
                if st.session_state.closed_start_time is None:
                    st.session_state.closed_start_time = time.time()
                duration = time.time() - st.session_state.closed_start_time
                
                if duration >= DROWSY_TIME:
                    status_ui.error("🚨 ALERT: DROWSY DETECTED!")
                else:
                    status_ui.warning("⚠️ Warning: Eyes Closing...")
            else:
                st.session_state.closed_start_time = None
                status_ui.success("✅ Driver Active")
            
            # 5. UI Text Overlay on Video
            status_text = "DROWSY!" if is_drowsy else "Eyes Open"
            status_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Convert to RGB for Streamlit display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, channels="RGB", use_column_width=True)
            
            # 6. Update Metrics
            ear_metric.metric("Avg EAR", f"{avg_ear:.3f}")
            timer_metric.metric("Closed Time", f"{duration:.1f}s")
            
            frame_count += 1
            time.sleep(0.01)  # Small delay to prevent overwhelming the UI

        cap.release()
else:
    status_ui.info("System Standby - Toggle Power to Start")
    frame_placeholder.image("https://via.placeholder.com/640x480?text=Camera+Off")

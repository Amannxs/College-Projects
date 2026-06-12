import streamlit as st
import cv2
import numpy as np
import time
import torch
from PIL import Image
# Apne purane app.py se functions import karein ya wahi logic yahan rakhein
# Maan lete hain aapke logic functions yahan define hain:
from app import face_mesh, compute_EAR, crop_single_eye, predict_eye, LEFT_EYE, RIGHT_EYE, stop_alarm, start_alarm

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
    ear_metric = st.metric("Avg EAR", "0.000")
    timer_metric = st.metric("Closed Time", "0.0s")
    st.divider()
    st.info("💡 Tip: Ensure good lighting on your face for better accuracy.")

# --- MAIN ENGINE ---
if run_app:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.error("Webcam access lost!")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Mediapipe Face Mesh
        results = face_mesh.process(rgb_frame)
        
        is_drowsy = False
        avg_ear = 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 2. EAR Logic
            left_ear = compute_EAR(landmarks, LEFT_EYE, w, h)
            right_ear = compute_EAR(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # 3. CNN Logic (Optional but integrated)
            left_crop, _ = crop_single_eye(frame, landmarks, LEFT_EYE, w, h)
            right_crop, _ = crop_single_eye(frame, landmarks, RIGHT_EYE, w, h)
            
            # Simple combined decision
            if avg_ear < EAR_THRESHOLD:
                is_drowsy = True
            
            # Draw Landmarks for Feedback
            for idx in LEFT_EYE + RIGHT_EYE:
                x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # 4. Timer & Alarm Logic
        duration = 0.0
        if is_drowsy:
            if st.session_state.closed_start_time is None:
                st.session_state.closed_start_time = time.time()
            duration = time.time() - st.session_state.closed_start_time
            
            if duration >= DROWSY_TIME:
                status_ui.error("🚨 ALERT: DROWSY DETECTED!")
                if not st.session_state.alarm_on:
                    start_alarm()
                    st.session_state.alarm_on = True
            else:
                status_ui.warning("⚠️ Warning: Eyes Closing...")
        else:
            st.session_state.closed_start_time = None
            status_ui.success("✅ Driver Active")
            if st.session_state.alarm_on:
                stop_alarm()
                st.session_state.alarm_on = False

        # 5. UI Updates
        ear_metric.metric("Avg EAR", f"{avg_ear:.3f}")
        timer_metric.metric("Closed Time", f"{duration:.1f}s")
        
        # Convert to RGB for Streamlit
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(display_frame, channels="RGB")

        if not run_app:
            break

    cap.release()
    stop_alarm()
else:
    status_ui.info("System Standby")
    frame_placeholder.image("https://via.placeholder.com/640x480.png?text=Camera+Off")
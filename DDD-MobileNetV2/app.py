# ==============================
# IMPORT LIBRARIES
# ==============================

# Fix for torchvision circular import on Windows
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import time
import torch
try:
    import mediapipe as mp
except Exception as e:
    print(f"Mediapipe import failed: {e}")
    mp = None
import winsound
import threading
import numpy as np

from PIL import Image
from collections import deque

import torch.nn as nn

# Import torchvision components with error handling
try:
    from torchvision import transforms, models
    TORCHVISION_OK = True
except Exception as e:
    print(f"Torchvision import issue: {e}")
    TORCHVISION_OK = False
    transforms = None
    models = None


# ==============================
# INITIALIZE ALARM
# ==============================

alarm_thread = None
alarm_stop_event = threading.Event()   # FIX: clean alarm stop control


# ==============================
# DEVICE
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")


# ==============================
# CLASS NAMES
# ==============================

CLASS_NAMES = ['Closed', 'Opened']


# ==============================
# LOAD MODEL
# ==============================

model = None
try:
    if models is not None:
        model = models.mobilenet_v2(pretrained=False)

        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, 2)
        )

        model_path = "model/driver_drowsiness_mobilenetv2.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model Loaded Successfully!")
        else:
            print(f"Warning: Model file not found at {model_path}")

        model = model.to(device)
        model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# ==============================
# IMAGE TRANSFORM
# ==============================

transform = None
if TORCHVISION_OK:
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    except Exception as e:
        print(f"Error creating transform: {e}")


# ==============================
# MEDIAPIPE FACE MESH
# ==============================

face_mesh = None
if mp is not None:
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        print(f"Error initializing Mediapipe FaceMesh: {e}")
        face_mesh = None
else:
    print("Warning: Mediapipe not initialized because import failed.")


# ==============================
# EYE LANDMARKS (kept exactly as yours)
# ==============================

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ==============================
# EAR FUNCTION  ← NEW
# ==============================

def compute_EAR(landmarks, eye_indices, w, h):
    pts = np.array([
        (landmarks[eye_indices[i]].x * w,
         landmarks[eye_indices[i]].y * h)
        for i in range(6)
    ])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    hz = np.linalg.norm(pts[0] - pts[3])
    if hz == 0:
        return 0.3  # safe default
    return (v1 + v2) / (2.0 * hz)


# ==============================
# SEPARATE EYE CROP  ← FIX: crop each eye individually
# ==============================

def crop_single_eye(frame, landmarks, eye_indices, w, h, pad=12):
    pts = np.array([
        (int(landmarks[eye_indices[i]].x * w),
         int(landmarks[eye_indices[i]].y * h))
        for i in range(6)
    ])
    x_min = max(pts[:, 0].min() - pad, 0)
    x_max = min(pts[:, 0].max() + pad, w)
    y_min = max(pts[:, 1].min() - pad, 0)
    y_max = min(pts[:, 1].max() + pad, h)
    crop = frame[y_min:y_max, x_min:x_max]
    return crop, (x_min, y_min, x_max, y_max)


# ==============================
# CNN PREDICT (reusable)  ← NEW
# ==============================

def predict_eye(eye_crop):
    """Returns (label, confidence) for a single eye crop."""
    if model is None or transform is None:
        return "Opened", 0.0

    eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(eye_rgb)
    tensor  = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return CLASS_NAMES[pred.item()], conf.item()


# ==============================
# ALARM CONTROL  ← FIX: proper stop/start
# ==============================

def play_alarm_loop(stop_event):
    while not stop_event.is_set():
        winsound.PlaySound("alarm.wav", winsound.SND_FILENAME)

def start_alarm():
    global alarm_thread, alarm_stop_event
    if alarm_thread is None or not alarm_thread.is_alive():
        alarm_stop_event.clear()
        alarm_thread = threading.Thread(
            target=play_alarm_loop,
            args=(alarm_stop_event,),
            daemon=True
        )
        alarm_thread.start()

def stop_alarm():
    global alarm_stop_event
    alarm_stop_event.set()
    winsound.PlaySound(None, winsound.SND_PURGE)


# ==============================
# VARIABLES
# ==============================

# FIX: separate history per eye instead of one merged buffer
left_history  = deque(maxlen=10)
right_history = deque(maxlen=10)

# FIX: EAR buffer for PERCLOS
ear_buffer = deque(maxlen=60)          # ~2 seconds at 30fps

closed_start_time = None
no_face_frames    = 0

DROWSY_THRESHOLD     = 1          # seconds eyes closed before alarm
CONFIDENCE_THRESHOLD = 0.80
EAR_THRESHOLD        = 0.25
CLOSED_VOTE_MIN      = 6              # FIX: lowered from 7 (out of 10), tunable
NO_FACE_RESET_LIMIT  = 10            # frames before clearing history on face loss

alarm_playing = False


# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    # ==============================
    # START CAMERA
    # ==============================
    # Using CAP_DSHOW to prevent Windows startup crashes
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Warning: Could not open webcam at index 0. Trying index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    print("Webcam Started... Press 'ESC' to exit.")

    try:
        while True:
            success, frame = cap.read()
            
            # FIX: Don't break if one frame fails, just try again
            if not success:
                print("Skipping empty frame...")
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ==============================
            # STEP 1: FACE DETECTION GATE
            # ==============================
            results = face_mesh.process(rgb_frame)

            if not results or not results.multi_face_landmarks:
                no_face_frames += 1

                if no_face_frames >= NO_FACE_RESET_LIMIT:
                    left_history.clear()
                    right_history.clear()
                    ear_buffer.clear()
                    closed_start_time = None

                    if alarm_playing:
                        stop_alarm()
                        alarm_playing = False

                cv2.putText(frame, "No Face Detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                cv2.imshow("Driver Drowsiness Detection", frame)

                if cv2.waitKey(1) == 27:
                    break
                continue 

            # ==============================
            # STEP 2: FACE CONFIRMED
            # ==============================
            no_face_frames = 0
            face_landmarks = results.multi_face_landmarks[0].landmark

            left_ear  = compute_EAR(face_landmarks, LEFT_EYE,  w, h)
            right_ear = compute_EAR(face_landmarks, RIGHT_EYE, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0
            ear_buffer.append(avg_ear)

            perclos = sum(1 for e in ear_buffer if e < EAR_THRESHOLD) / max(len(ear_buffer), 1)

            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(face_landmarks[idx].x * w)
                y = int(face_landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # ==============================
            # STEP 3: CROP & PREDICT
            # ==============================
            left_crop,  left_box  = crop_single_eye(frame, face_landmarks, LEFT_EYE,  w, h)
            right_crop, right_box = crop_single_eye(frame, face_landmarks, RIGHT_EYE, w, h)

            left_pred = right_pred = "Opened"
            left_conf = right_conf = 0.0

            if left_crop is not None and left_crop.size != 0:
                left_pred, left_conf = predict_eye(left_crop)
            if right_crop is not None and right_crop.size != 0:
                right_pred, right_conf = predict_eye(right_crop)

            if left_conf >= CONFIDENCE_THRESHOLD:
                left_history.append(left_pred)
            if right_conf >= CONFIDENCE_THRESHOLD:
                right_history.append(right_pred)

            def majority(hist):
                if len(hist) == 0: return "Opened"
                return max(set(hist), key=list(hist).count)

            left_final  = majority(left_history)
            right_final = majority(right_history)
            both_closed = (left_final == "Closed" and right_final == "Closed")

            for box, pred, conf, label in [(left_box, left_pred, left_conf, "L"), (right_box, right_pred, right_conf, "R")]:
                x1, y1, x2, y2 = box
                color = (0, 0, 255) if pred == "Closed" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}:{pred} {conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            # ==============================
            # STEP 4: DECISION
            # ==============================
            ear_says_open = avg_ear > (EAR_THRESHOLD + 0.05)
            perclos_alert = perclos > 0.70

            if ear_says_open:
                is_drowsy = False
            elif both_closed or perclos_alert:
                is_drowsy = True
            else:
                is_drowsy = False

            # ==============================
            # STEP 5: TIMER & ALARM
            # ==============================
            if is_drowsy:
                status_text, status_color = "DROWSY!", (0, 0, 255)
                if closed_start_time is None: closed_start_time = time.time()
                closed_duration = time.time() - closed_start_time

                cv2.putText(frame, f"Eyes Closed: {closed_duration:.1f}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if closed_duration >= DROWSY_THRESHOLD:
                    cv2.putText(frame, "WAKE UP!", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if not alarm_playing:
                        start_alarm()
                        alarm_playing = True
            else:
                closed_start_time = None
                status_text, status_color = "Eyes Open", (0, 255, 0)
                if alarm_playing:
                    stop_alarm()
                    alarm_playing = False

            # UI Overlays
            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
            cv2.putText(frame, f"EAR: {avg_ear:.3f} | PERCLOS: {perclos:.0%}", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Driver Drowsiness Detection", frame)
            if cv2.waitKey(1) == 27: break

    except Exception as e:
        print(f"\n[PROGRAM CRASHED]: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("Releasing Resources...")
        cap.release()
        cv2.destroyAllWindows()
        stop_alarm()

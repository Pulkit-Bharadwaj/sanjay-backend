import os
import sys

# 1. FORCE CORE ENVIRONMENT VARIABLES
os.environ["TF_USE_LEGACY_KERAS"] = "1"       
os.environ["TF_AUTOGRAPH_IMPLEMENTATION"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'      
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     

try:
    import tensorflow as tf
    if hasattr(tf, "__internal__") and not hasattr(tf.__internal__, "register_load_context_function"):
        tf.__internal__.register_load_context_function = lambda x: None
except Exception:
    pass

import tempfile
import numpy as np
import streamlit as st
from PIL import Image
import cv2
from deepface import DeepFace

st.set_page_config(page_title="Sanjay: AI Re-Identification", page_icon="🔍")
st.title("🔍 Sanjay - AI Based Re-identification Model")

# Onboarding Instructions Panel
with st.expander("ℹ️ First Time User? Click here for 3 simple steps to get started"):
    st.write("Welcome! This app uses smart face recognition to scan videos and locate a specific person automatically.")
    st.markdown("""
    1. **Upload a clear photo** of the person.
    2. **Upload your video file**.
    3. **Note:** If the person isn't found, try a photo where the person is facing the camera directly.
    """)

# Core Vector Embedding Function
def get_embedding(image_array):
    try:
        result = DeepFace.represent(
            img_path=image_array,
            model_name='ArcFace',
            detector_backend='skip',
            enforce_detection=False
        )
        if result and len(result) > 0:
            return np.array(result[0]['embedding'])
        return None
    except Exception:
        return None

def cosine_similarity(e1, e2):
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)
    return float(np.dot(e1, e2))

ref_file = st.file_uploader("📷 Upload Reference Image", type=['jpg','jpeg','png'])
video_file = st.file_uploader("🎥 Upload Video Clip", type=['mp4','avi','mov'])

if ref_file and video_file:
    # ADDED: Sensitivity Slider
    sensitivity = st.slider("Match Sensitivity", 0.1, 0.9, 0.35, help="Lower value makes it easier to match.")
    
    if st.button("🚀 Identify Person"):
        try:
            from PIL import ImageOps
            pil_img = ImageOps.exif_transpose(Image.open(ref_file))
        except Exception:
            pil_img = Image.open(ref_file)
            
        pil_img = pil_img.convert('RGB')
        ref_img_rgb = np.array(pil_img)

        with st.spinner("Analyzing reference..."):
            ref_emb = get_embedding(ref_img_rgb)

        if ref_emb is None:
            st.error("❌ No face found in reference image.")
        else:
            st.success("✅ Reference vectorized!")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path = os.path.join(tempfile.gettempdir(), "out.mp4")
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            frame_num = 0
            timestamps = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                if frame_num % 5 == 0:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        faces = DeepFace.extract_faces(frame_rgb, detector_backend='mediapipe', enforce_detection=False)
                        
                        for face in faces:
                            if face['confidence'] > 0.3: # Lowered confidence requirement
                                x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
                                emb = get_embedding(frame_rgb[y:y+h, x:x+w])
                                
                                if emb is not None and cosine_similarity(ref_emb, emb) >= sensitivity:
                                    timestamps.append(round(frame_num / fps, 2))
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    except: pass

                out.write(frame)
                frame_num += 1

            cap.release()
            out.release()
            
            if timestamps:
                st.success("✅ Match Found!")
                st.write("### 🕐 Appears at:", ", ".join([f"{t}s" for t in sorted(set(timestamps))]))
                with open(out_path, 'rb') as f:
                    st.download_button("⬇️ Download Result", f, "output.mp4", "video/mp4")
            else:
                st.warning("⚠️ No match found. Try uploading a clearer reference photo or lowering the 'Match Sensitivity' slider.")
import os
import sys

# 1. ENVIRONMENT STABILIZATION
os.environ["TF_USE_LEGACY_KERAS"] = "1"       
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'      

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

# Page Setup
st.set_page_config(page_title="Sanjay: AI Re-Identification", page_icon="🔍")
st.title("🔍 Sanjay - AI Based Re-identification Model")

# Instructions
with st.expander("ℹ️ How to get better matches"):
    st.markdown("""
    * **Sensitivity:** If no match is found, your reference image and video face must be very similar.
    * **Lighting:** Ensure the face in the video has similar lighting to your reference photo.
    * **Quality:** Higher resolution clips produce significantly better biometric vector results.
    """)

# Core Engine
def get_embedding(img_input):
    try:
        # Use 'skip' for high-precision vector mapping
        result = DeepFace.represent(
            img_path=img_input,
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

# UI
ref_file = st.file_uploader("📷 Upload Reference Image", type=['jpg','jpeg','png'])
video_file = st.file_uploader("🎥 Upload Video Clip", type=['mp4','avi','mov'])

if ref_file and video_file:
    if st.button("🚀 Identify Person"):
        
        # Load & Normalize
        pil_img = Image.open(ref_file).convert('RGB')
        ref_img_rgb = np.array(pil_img)

        with st.spinner("Analyzing identity..."):
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
            
            out_path = os.path.join(tempfile.gettempdir(), "output.mp4")
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                                  (int(cap.get(3)), int(cap.get(4))))

            timestamps = []
            frame_num = 0
            progress = st.progress(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Analyze every 3rd frame (faster but high precision)
                if frame_num % 3 == 0:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Detection using mediapipe
                        faces = DeepFace.extract_faces(frame_rgb, detector_backend='mediapipe', enforce_detection=False)
                        
                        for f in faces:
                            if f['confidence'] > 0.2: # Lowered confidence requirement
                                x, y, w, h = f['facial_area']['x'], f['facial_area']['y'], f['facial_area']['w'], f['facial_area']['h']
                                face_crop = frame_rgb[y:y+h, x:x+w]
                                
                                emb = get_embedding(face_crop)
                                if emb is not None:
                                    score = cosine_similarity(ref_emb, emb)
                                    if score >= 0.35: # Lowered match threshold for higher leniency
                                        timestamps.append(round(frame_num / fps, 2))
                                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    except: pass

                out.write(frame)
                frame_num += 1
                progress.progress(min(frame_num / int(cap.get(7)), 1.0))

            cap.release()
            out.release()
            
            if timestamps:
                st.success("✅ Match Found!")
                st.write(f"### 🕐 Found at: {', '.join([str(t)+'s' for t in sorted(set(timestamps))])}")
                with open(out_path, 'rb') as f:
                    st.download_button("⬇️ Download Video", f, "output.mp4", "video/mp4")
            else:
                st.warning("⚠️ No match found. Try a different video or photo.")
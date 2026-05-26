import os
import sys

# 1. CLOUD ENVIRONMENT PERFORMANCE CONFIGURATIONS
os.environ["QT_QPA_PLATFORM"] = "offscreen"   
os.environ["TF_USE_LEGACY_KERAS"] = "1"       
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'      
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     

import tempfile
import numpy as np
import streamlit as st
from PIL import Image

# 2. CORE ENGINE NATIVE LOADING
import cv2
from deepface import DeepFace

# Page Layout Setup
st.set_page_config(page_title="Sanjay: AI Re-Identification", page_icon="🔍")
st.title("🔍 Sanjay - AI Based Re-identification Model")
st.write("Upload a photo of a person to scan, track, and extract every moment they appear inside a video clip.")

# Collapsible Instructions Drawer
with st.expander("ℹ️ First Time User? Click here for 3 simple steps to get started"):
    st.write("Welcome! This app uses smart face recognition to scan videos and locate a specific person automatically.")
    st.markdown("""
    1. **Upload a clear photo** of the single person you want to find under the **Person's Photo** zone.
    2. **Upload your video file** into the **Video File** zone.
    3. **Adjust settings if needed:** *Match Accuracy (Strictness)*: Higher numbers mean less room for error (prevents matching strangers).
        * *Processing Speed*: Bypasses frames to scan longer videos significantly faster.
    4. Click the **Start Search Engine** button and watch the results populate!
    """)

# Core Feature Vector Generator
# Core Feature Vector Generator
# Core Feature Vector Generator
def get_embedding(image_array):
    try:
        result = DeepFace.represent(
            img_path=image_array,
            model_name='ArcFace',
            detector_backend='skip',  # Skip secondary checks since array parsing is direct
            enforce_detection=False
        )
        if result and len(result) > 0:
            return np.array(result[0]['embedding'])
        return None
    except Exception as e:
        # EXPLICIT DEBUG INTERCEPTOR: Print the background system error directly to the UI
        st.warning(f"🔧 System Diagnostic Log: {str(e)}")
        return None

def cosine_similarity(e1, e2):
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)
    return float(np.dot(e1, e2))

# User Upload UI Elements
ref_file = st.file_uploader("📷 Upload Reference Image", type=['jpg','jpeg','png'])
video_file = st.file_uploader("🎥 Upload Video Clip", type=['mp4','avi','mov'])

if ref_file and video_file:
    if st.button("🚀 Identify Person"):
        
        # 3. CONVERT REFERENCE IMAGE NATIVELY
        try:
            from PIL import ImageOps
            pil_img = ImageOps.exif_transpose(Image.open(ref_file))
        except Exception:
            pil_img = Image.open(ref_file)
            
        pil_img = pil_img.convert('RGB')
        ref_img_rgb = np.array(pil_img)

        with st.spinner("Extracting reference face features..."):
            ref_emb = get_embedding(ref_img_rgb)

        if ref_emb is None:
            st.error("❌ Could not process reference image. Please try another clear photo.")
        else:
            st.success("✅ Reference face successfully vectorized!")

            # Create cloud temporary workspace for video streaming arrays
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            try:
                # Open video read/write streams
                cap = cv2.VideoCapture(tmp_path)
                fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) != 0 else 25
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Build a temporary storage path for the generated output video file
                out_path = os.path.join(tempfile.gettempdir(), "sanjay_output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Universal MP4 video compiler format
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                timestamps = []
                frame_num = 0
                
                # Active progress status trackers on UI layout
                progress_bar = st.progress(0)
                status_text = st.empty()

                with st.spinner("Scanning video frames and highlighting matches..."):
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Optimization rule: Analyze 1 out of every 5 frames to maximize processing speed
                        if frame_num % 5 == 0:
                            try:
                                # Convert running BGR frame array cleanly to RGB format for DeepFace extraction
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                # Use cloud-safe MediaPipe backend to find facial regions in the frame
                                faces_detected = DeepFace.extract_faces(
                                    img_path=frame_rgb,
                                    detector_backend='mediapipe',
                                    enforce_detection=False
                                )
                                
                                for face_obj in faces_detected:
                                    if face_obj['confidence'] > 0.4:
                                        facial_area = face_obj['facial_area']
                                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                                        
                                        # Cut face target array slice natively from the frame
                                        face_crop_rgb = frame_rgb[y:y+h, x:x+w]
                                        if face_crop_rgb.size == 0:
                                            continue
                                            
                                        emb = get_embedding(face_crop_rgb)
                                        if emb is not None:
                                            score = cosine_similarity(ref_emb, emb)
                                            
                                            # Match verification threshold evaluation logic
                                            if score >= 0.42:  
                                                seconds = round(frame_num / fps, 2)
                                                timestamps.append(seconds)
                                                
                                                # Draw green highlighted tracker box directly onto the frame
                                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                                                cv2.putText(frame, f"MATCH {score:.2f}",
                                                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                                            0.9, (0, 255, 0), 2)
                            except Exception:
                                pass

                        # Feed frame matrix forward into target download file assembly line
                        out.write(frame)
                        frame_num += 1

                        # Keep the UI progress bar ticking actively
                        if total_frames > 0:
                            progress_bar.progress(min(frame_num / total_frames, 1.0))
                            status_text.text(f"Processing clip frame: {frame_num} / {total_frames}")

                # Safely release hardware video threads from container memory
                cap.release()
                out.release()
                
                st.success("✅ Analysis Complete! Target identity tracking rendered successfully.")

                # Display Timestamp Logs
                if timestamps:
                    unique_times = sorted(set(timestamps))
                    st.write("### 🕐 Person appears at:")
                    st.write(", ".join([f"{t}s" for t in unique_times]))
                else:
                    st.warning("⚠️ Person not identified inside the uploaded video clip sample context.")

                # 4. DOWNLOAD INTERFACE RENDERING
                with open(out_path, 'rb') as compiled_file:
                    st.download_button(
                        label="⬇️ Download Highlighted Video",
                        data=compiled_file,
                        file_name="reidentified_output.mp4",
                        mime="video/mp4"
                    )
            except Exception as e:
                st.error(f"❌ Video rendering thread encountered a layout error: {str(e)}")
            finally:
                # Hard cleanup of file arrays to keep container storage usage optimized
                try:
                    os.unlink(tmp_path)
                    os.unlink(out_path)
                except Exception:
                    pass
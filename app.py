import os
import sys

# 1. Force OpenCV and QT to run in offscreen headless mode
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# 2. Wrap the imports inside a completely protected block
try:
    import cv2
    import numpy as np
except ImportError:
    # If a secondary binary is missing, force Python to mock the cv2 module name
    import sys
    from types import ModuleType
    sys.modules['cv2'] = ModuleType('cv2')
    import cv2

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from deepface import DeepFace

st.set_page_config(page_title="Sanjay: AI Re-Identification", page_icon="🔍")
st.title("🔍 Sanjay: AI Re-Identification")
st.write("Upload a reference photo and a video to identify the person.")

def get_embedding(image_bgr):
    try:
        result = DeepFace.represent(
            img_path=image_bgr,
            model_name='Facenet',
            enforce_detection=False
        )
        if result:
            return np.array(result[0]['embedding'])
        return None
    except Exception as e:
        return None

def cosine_similarity(e1, e2):
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)
    return float(np.dot(e1, e2))

ref_file = st.file_uploader("📷 Upload Reference Image", type=['jpg','jpeg','png'])
video_file = st.file_uploader("🎥 Upload Video Clip", type=['mp4','avi','mov'])

if ref_file and video_file:
    if st.button("🚀 Identify Person"):
        ref_bytes = np.asarray(bytearray(ref_file.read()), dtype=np.uint8)
        ref_img = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Getting reference face embedding..."):
            ref_emb = get_embedding(ref_img)

        if ref_emb is None:
            st.error("❌ No face found in reference image. Try a clearer photo.")
        else:
            st.success("✅ Reference face detected!")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 25

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path = tmp_path.replace('.mp4', '_output.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            timestamps = []
            frame_num = 0
            progress = st.progress(0)
            status = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % 5 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                    for (x, y, w, h) in faces:
                        face_crop = frame[y:y+h, x:x+w]
                        if face_crop.size == 0:
                            continue
                        emb = get_embedding(face_crop)
                        if emb is not None:
                            score = cosine_similarity(ref_emb, emb)
                            if score >= 0.45:
                                seconds = round(frame_num / fps, 2)
                                timestamps.append(seconds)
                                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
                                cv2.putText(frame, f"MATCH {score:.2f}",
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0,255,0), 2)

                out.write(frame)
                frame_num += 1

                if total_frames > 0:
                    progress.progress(min(frame_num / total_frames, 1.0))
                    status.text(f"Processing frame {frame_num}/{total_frames}")

            cap.release()
            out.release()
            os.unlink(tmp_path)

            st.success("✅ Processing Complete!")

            if timestamps:
                unique_times = sorted(set(timestamps))
                st.write("### 🕐 Person appears at:")
                st.write(", ".join([f"{t}s" for t in unique_times]))
            else:
                st.warning("⚠️ Person not found in video.")

            with open(out_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=f,
                    file_name="output.mp4",
                    mime="video/mp4"
                )
            os.unlink(out_path)
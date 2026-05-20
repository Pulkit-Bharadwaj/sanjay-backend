import cv2
from embedder import get_embedding
from matcher import is_match

def process_video(video_path, ref_embedding, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    timestamps = []
    frame_num = 0
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 5th frame for speed
        if frame_num % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue
                emb = get_embedding(face_crop)
                if emb is not None:
                    matched, score = is_match(ref_embedding, emb)
                    if matched:
                        seconds = round(frame_num / fps, 2)
                        timestamps.append(seconds)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        cv2.putText(frame, f"MATCH {score:.2f}",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    return list(set(timestamps))
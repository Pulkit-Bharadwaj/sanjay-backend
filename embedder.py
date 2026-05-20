from deepface import DeepFace
import numpy as np
import cv2

def get_embedding(image_bgr):
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = DeepFace.represent(
            img_path=image_rgb,
            model_name='Facenet',
            enforce_detection=False
        )
        if result:
            return np.array(result[0]['embedding'])
        return None
    except Exception as e:
        print(f"Embedding error: {e}")
        return None
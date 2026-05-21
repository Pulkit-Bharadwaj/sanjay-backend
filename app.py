from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2

# Pre-load model when server starts (not on first request)
print("Pre-loading FaceNet model...")
from embedder import get_embedding
import numpy as np
dummy = np.zeros((100, 100, 3), dtype=np.uint8)
get_embedding(dummy)
print("Model loaded successfully!")

app = Flask(__name__)
CORS(app, origins="*")

UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return jsonify({'status': 'Sanjay backend is running!'})

@app.route('/process', methods=['POST', 'OPTIONS'])
def process():
    if request.method == 'OPTIONS':
        return '', 200

    if 'reference' not in request.files or 'video' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    ref_image = request.files['reference']
    video = request.files['video']

    ref_path = os.path.join(UPLOAD_FOLDER, 'ref.jpg')
    vid_path = os.path.join(UPLOAD_FOLDER, 'input.mp4')
    out_path = os.path.join(OUTPUT_FOLDER, 'output.mp4')

    ref_image.save(ref_path)
    video.save(vid_path)

    ref_frame = cv2.imread(ref_path)
    if ref_frame is None:
        return jsonify({'error': 'Could not read reference image'}), 400

    ref_emb = get_embedding(ref_frame)
    if ref_emb is None:
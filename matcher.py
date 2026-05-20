import numpy as np

def cosine_similarity(emb1, emb2):
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1, emb2))

def is_match(emb1, emb2, threshold=0.45):
    score = cosine_similarity(emb1, emb2)
    return score >= threshold, score
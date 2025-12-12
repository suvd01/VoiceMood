# face_emotion.py
import cv2
from deepface import DeepFace
from typing import Tuple, Optional


def analyze_face_frame(frame) -> Tuple[Optional[str], float]:
    
    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
        )

        
        if isinstance(result, list):
            result = result[0]

        dominant_emotion = result.get("dominant_emotion", None)
        emotion_scores = result.get("emotion", {})
        if dominant_emotion and emotion_scores:
            conf = float(emotion_scores.get(dominant_emotion, 0.0))
        else:
            conf = 0.0

        return dominant_emotion, conf

    except Exception as e:
        return None, 0.0

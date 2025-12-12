# emotion_utils.py
from typing import Dict

EMOJI_MAP = {
    "happy": ("😀", "#FFD54F", "행복"),
    "sad": ("😢", "#64B5F6", "슬픔"),
    "angry": ("😡", "#E57373", "분노"),
    "fear": ("😨", "#9575CD", "두려움"),
    "surprise": ("😮", "#FFB74D", "놀람"),
    "neutral": ("😐", "#B0BEC5", "중립"),
    "disgust": ("🤢", "#81C784", "역겨움"),
}

DEFAULT_EMOTION = "neutral"


def normalize_emotion_name(name: str) -> str:
   
    if not name:
        return DEFAULT_EMOTION

    n = name.lower()

    if "happy" in n or "joy" in n:
        return "happy"
    if "sad" in n:
        return "sad"
    if "angry" in n or "mad" in n:
        return "angry"
    if "fear" in n or "scared" in n:
        return "fear"
    if "surprise" in n or "surprised" in n:
        return "surprise"
    if "disgust" in n:
        return "disgust"
    if "neutral" in n or "calm" in n:
        return "neutral"

    return DEFAULT_EMOTION


def get_emotion_style(emotion: str):
  
    e = normalize_emotion_name(emotion)
    emoji, color, ko = EMOJI_MAP.get(e, EMOJI_MAP[DEFAULT_EMOTION])
    return e, emoji, color, ko


def merge_emotions(
    face_emotion: str,
    face_conf: float,
    voice_emotion: str,
    voice_conf: float,
) -> str:
    

    f = normalize_emotion_name(face_emotion) if face_emotion else None
    v = normalize_emotion_name(voice_emotion) if voice_emotion else None

    if f and v:
        if f == v:
            return f
        
        if voice_conf >= face_conf:
            return v
        else:
            return f

    if f:
        return f
    if v:
        return v

    return DEFAULT_EMOTION

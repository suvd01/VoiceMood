# voice_emotion.py
import numpy as np
import librosa
import sounddevice as sd
import pickle
from scipy.signal import butter, lfilter


with open("voice_emotion_model.pkl", "rb") as f:
    VOICE_MODEL = pickle.load(f)

IDX2LABEL = {
    0: "angry",
    1: "happy",
    2: "neutral",
    3: "sad"
}


def extract_mfcc(audio, sr, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean


def extract_mfcc_from_file(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    return extract_mfcc(y, sr, n_mfcc=n_mfcc)


def predict_emotion_from_features(feat_vec: np.ndarray):
    x = feat_vec.reshape(1, -1)

    proba = VOICE_MODEL.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    emotion = IDX2LABEL[idx]
    confidence = float(proba[idx])

    prob_dict = {
        IDX2LABEL[i]: float(p)
        for i, p in enumerate(proba)
    }

    return emotion, confidence, prob_dict


def analyze_voice_file(audio_path: str):
    try:
        feat = extract_mfcc_from_file(audio_path, n_mfcc=40)
        emotion, conf, prob = predict_emotion_from_features(feat)
        return emotion, conf, prob
    except Exception as e:
        print("[WARN] Voice file analysis failed:", e)
        return "neutral", 0.0, {}




def butter_highpass(cutoff=80, fs=16000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def highpass_filter(data, cutoff=80, fs=16000):
    b, a = butter_highpass(cutoff, fs)
    return lfilter(b, a, data)


def analyze_realtime_voice(duration=2, sr=16000):
    print("🎙 실시간 음성 녹음 중…")

   
    audio = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    audio = audio.flatten()

    
    if np.max(np.abs(audio)) < 0.02:
        return "neutral", 0.1, {}

    
    audio = highpass_filter(audio, cutoff=80, fs=sr)

   
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    feat_vec = np.mean(mfcc.T, axis=0)

    
    emotion, conf, prob = predict_emotion_from_features(feat_vec)

    return emotion, conf, prob


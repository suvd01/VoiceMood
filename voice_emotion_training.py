import os
import pickle
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

emotion_labels = {
    "angry": 0,
    "happy": 1,
    "neutral": 2,
    "sad": 3
}

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def load_dataset(folder="voice_dataset"):
    X = []
    y = []

    for emotion, label in emotion_labels.items():
        file_path = os.path.join(folder, f"{emotion}.wav")
        if os.path.exists(file_path):
            print(f"Loading {file_path} ...")
            features = extract_mfcc(file_path)
            X.append(features)
            y.append(label)
        else:
            print(f"❌ Missing file: {file_path}")

    return np.array(X), np.array(y)

def train_model():
    X, y = load_dataset()

    if len(X) < 2:
        print("❌ Not enough audio files for training.")
        return

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    with open("voice_emotion_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\n🎉 Training complete! Model saved as voice_emotion_model.pkl")

if __name__ == "__main__":
    train_model()

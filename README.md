# VoiceMood
음성-얼굴 기반 감정 인식 시스템 (VoiceMood project)
추천하는 python:
Python 3.10 ~ 3.11

Installation:
pip install -r requirements.txt

requirements 안에:
numpy
opencv-python
pyqt5
tensorflow
librosa
sounddevice
scikit-learn
deepface 이라는 필요힌 것들이 다 있음.

How to run:
python main.py

시스템 동작 방식:
(1) 얼굴 감정 인식 부분
OpenCV를 이용해 웹캠을 활성화합니다.
DeepFace 모델을 사용하여 다음 감정을 분석합니다. Happy, sad, angry, neutral 분석된 감정은 UI 화면에 이모지 + 텍스트 형태로 즉시 표시됩니다.
(2) 음성 감정 인식 부분
SoundDevice로 마이크 입력을 받아 짧은 음성을 녹음합니다.
Librosa를 이용하여 MFCC 특징을 추출합니다.
RandomForest 기반 모델(voice_emotion_model.pkl)로 감정을 분류합니다.
voice_emotion_model.pkl 이 파일은 사전에 학습된 음성 감정 모델입니다. 프로젝트 실행 시 자동으로 로드됩니다.
(3) 감정 통합 
얼굴 감정과 음성 감정이 서로 다를 경우 다음 기준으로 최종 감정을 결정합니다:
두 모델의 confidence score를 비교합니다.
신뢰도가 더 높은 결과에 가중치를 줍니다. 두 결과의 차이가 너무 크면 안전하게 neutral로 처리합니다. 이 로직은 merge_emotions() 함수 내부에서 구현되어 있습니다.

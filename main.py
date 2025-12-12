# main.py
import sys
import cv2

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PyQt5.QtCore import QTimer, Qt

from face_emotion import analyze_face_frame
from voice_emotion import analyze_voice_file, analyze_realtime_voice
from emotion_utils import get_emotion_style, merge_emotions


class VoiceMoodApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("VoiceMood 🧠 감정 인식 AI 앱")
        self.resize(900, 600)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.current_face_emotion = None
        self.current_face_conf = 0.0
        self.current_voice_emotion = None
        self.current_voice_conf = 0.0

        self.setup_ui()

    def setup_ui(self):
        
        self.camera_label = QLabel("웹캠이 여기에 표시됩니다.")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(
            "background-color: #ECEFF1; border: 1px solid #B0BEC5;"
        )
        self.camera_label.setMinimumSize(400, 300)

        self.btn_start_cam = QPushButton("얼굴 감정 인식 시작")
        self.btn_stop_cam = QPushButton("중지")
        self.btn_start_cam.clicked.connect(self.start_camera)
        self.btn_stop_cam.clicked.connect(self.stop_camera)

        cam_btn_layout = QHBoxLayout()
        cam_btn_layout.addWidget(self.btn_start_cam)
        cam_btn_layout.addWidget(self.btn_stop_cam)

        
        self.btn_voice = QPushButton("음성 파일 선택 후 감정 분석")
        self.btn_voice.clicked.connect(self.select_voice_file)

        self.btn_realtime = QPushButton("🎙 실시간 음성 분석")
        self.btn_realtime.clicked.connect(self.start_realtime_voice)

        self.voice_result_label = QLabel("음성 감정: -")
        self.voice_result_label.setFont(QFont("Arial", 12))

        self.face_result_label = QLabel("얼굴 감정: -")
        self.face_result_label.setFont(QFont("Arial", 12))

        self.final_result_label = QLabel("최종 감정 결과")
        self.final_result_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.final_result_label.setAlignment(Qt.AlignCenter)
        self.final_result_label.setAutoFillBackground(True)
        self.update_final_display("neutral")

        
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.camera_label)
        left_layout.addLayout(cam_btn_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.btn_voice)
        right_layout.addWidget(self.btn_realtime)
        right_layout.addWidget(self.voice_result_label)
        right_layout.addWidget(self.face_result_label)
        right_layout.addStretch()
        right_layout.addWidget(self.final_result_label)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 2)

        self.setLayout(main_layout)

  
    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.timer.isActive():
            self.timer.start(50)

    def stop_camera(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_label.setPixmap(QPixmap())

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        emotion, conf = analyze_face_frame(rgb)
        self.current_face_emotion = emotion
        self.current_face_conf = conf

        e, emoji, color, ko = get_emotion_style(emotion)
        self.face_result_label.setText(
            f"얼굴 감정: {ko} {emoji} (신뢰도: {conf:.2f})"
        )

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.camera_label.setPixmap(pix)

        self.update_final_emotion()

    
    def select_voice_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "음성 파일 선택",
            "",
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)",
        )
        if not file_path:
            return

        voice_emotion, conf, prob = analyze_voice_file(file_path)
        self.current_voice_emotion = voice_emotion
        self.current_voice_conf = conf

        e, emoji, color, ko = get_emotion_style(voice_emotion)
        self.voice_result_label.setText(
            f"음성 감정: {ko} {emoji} (신뢰도: {conf:.2f})"
        )

        self.update_final_emotion()

    
    def start_realtime_voice(self):
        try:
            emotion, conf, prob = analyze_realtime_voice()
            self.current_voice_emotion = emotion
            self.current_voice_conf = conf

            e, emoji, color, ko = get_emotion_style(emotion)
            self.voice_result_label.setText(
                f"음성 감정: {ko} {emoji} (신뢰도: {conf:.2f})"
            )

            self.update_final_emotion()
        except Exception as e:
            print("실시간 음성 분석 오류:", e)

    
    def update_final_display(self, emotion):
        e, emoji, color, ko = get_emotion_style(emotion)
        pal = self.final_result_label.palette()
        pal.setColor(QPalette.Window, QColor(color))
        self.final_result_label.setPalette(pal)
        self.final_result_label.setText(f"{emoji} {ko}")

    def update_final_emotion(self):
        final = merge_emotions(
            self.current_face_emotion,
            self.current_face_conf,
            self.current_voice_emotion,
            self.current_voice_conf,
        )
        self.update_final_display(final)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VoiceMoodApp()
    win.show()
    sys.exit(app.exec_())

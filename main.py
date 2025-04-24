import subprocess
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QWidget, QCheckBox, QSlider, QHBoxLayout, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer
import noisereduce as nr
from pyrnnoise import RNNoise

class WaveformWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.audio_data = np.zeros(480)
        self.setMinimumHeight(100)

        # Timer to trigger repaint ~30fps
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(33)

    def update_waveform(self, data):
        self.audio_data = data.copy()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        pen = QPen(QColor(0,255,0))
        pen.setWidth(2)
        painter.setPen(pen)

        mid_y = self.height() / 2
        scale_x = self.width() / len(self.audio_data)
        scale_y = self.height() / 2

        for i in range(len(self.audio_data) - 1):
            x1 = i * scale_x
            y1 = mid_y - self.audio_data[i] * scale_y
            x2 = (i+1) * scale_x
            y2 = mid_y - self.audio_data[i+1] * scale_y
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))


class AudioFilterApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RNNoise Real-Time Filter")
        self.setGeometry(100, 100, 300, 150)

        self.process = None
        self.stream = None

        # Filter control states
        self.filter_enabled = True
        self.input_gain = 1.0
        self.output_volume = 1.0

        # UI Elements
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.status_label = QLabel("Status: Stopped")

        self.filter_checkbox = QCheckBox("Enable RNNoise Filter")
        self.filter_checkbox.setChecked(True)
        self.filter_checkbox.stateChanged.connect(self.toggle_filter)

        self.input_slider = QSlider(Qt.Horizontal)
        self.input_slider.setRange(0, 200)  # 0% to 200%
        self.input_slider.setValue(100)
        self.input_slider.valueChanged.connect(self.update_input_gain)

        self.output_slider = QSlider(Qt.Horizontal)
        self.output_slider.setRange(0, 200)
        self.output_slider.setValue(100)
        self.output_slider.valueChanged.connect(self.update_output_volume)

        self.input_label = QLabel("Input Gain: 100%")
        self.output_label = QLabel("Output Volume: 100%")

        self.waveform = WaveformWidget()
        self.rnnoise_inst = RNNoise(sample_rate=48000)

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Live Waveform:"))
        layout.addWidget(self.waveform)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.filter_checkbox)
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_slider)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def toggle_filter(self, state):
        self.filter_enabled = (state == Qt.Checked)

    def update_input_gain(self, value):
        self.input_gain = value / 100.0
        self.input_label.setText(f"Input Gain: {value}%")

    def update_output_volume(self, value):
        self.output_volume = value / 100.0
        self.output_label.setText(f"Output Volume: {value}%")

    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)

        try:
            # Flatten and apply gain
            input_data = indata[:, 0]
            processed = input_data * self.input_gain

            # Update live waveform
            self.waveform.update_waveform(processed[:480])

            if self.filter_enabled:
                int_data = (processed * 32768).astype(np.int16)
                denoised = np.zeros_like(int_data)

                for i in range(len(int_data)):
                    denoised[i] = self.rnnoise_inst.process_frame(int_data[i])

                out_float32 = denoised.astype(np.float32) / 32768.0
            else:
                # No filter
                out_float32 = processed

            out_float32 = np.asarray(out_float32).flatten()
            out_float32 *= self.output_volume
            out_float32 = np.clip(out_float32, -1.0, 1.0)
            outdata[:] = out_float32.reshape(-1, 1)

        except Exception as e:
            print(f"Audio callback error: {e}")
            outdata.fill(0)


    def start_stream(self):
        if self.stream is None:
            try:
                self.process = subprocess.Popen(
                    ['rnnoise'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0
                )

                self.stream = sd.Stream(
                    samplerate=48000,
                    device=None,
                    channels=1,
                    blocksize=480,
                    dtype='float32',
                    callback=self.audio_callback
                )
                self.stream.start()
                self.status_label.setText("Status: Running")
            except Exception as e:
                self.status_label.setText(f"Error: {e}")
                self.stop_stream()

    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.process:
            try:
                self.process.stdin.close()
                self.process.stdout.close()
            except:
                pass
            self.process.terminate()
            self.process.wait()
            self.process = None

        self.status_label.setText("Status: Stopped")

    def process_with_rnnoise(self, input_int16):
        process = subprocess.Popen(
            ['rnnoise'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )
        process.stdin.write(input_int16.tobytes())
        process.stdin.flush()
        output_bytes = process.stdout.read(len(input_int16) * 2)
        process.terminate()
        return output_bytes



if __name__ == "__main__":
    app = QApplication([])
    window = AudioFilterApp()
    window.show()
    app.exec_()
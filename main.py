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

        self.input_devices = [
            device for device in sd.query_devices() if device['max_input_channels'] > 0
        ]
        self.device_selector = QComboBox()
        for i, device in enumerate(self.input_devices):
            self.device_selector.addItem(f"{i}:{device['name']}")
        
        self.device_index = 0
        self.device_selector.currentIndexChanged.connect(self.select_device)

        self.waveform = WaveformWidget()

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
        layout.addWidget(QLabel("Select Microphone:"))
        layout.addWidget(self.device_selector)

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
    
    def select_device(self, index):
        self.device_index = self.input_devices[index]['index']

    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)

        # Apply input gain
        processed = indata[:, 0] * self.input_gain
        self.waveform.update_waveform(processed[:480])

        # Convert to int16 for rnnoise-cli
        int_data = (indata[:, 0] * 32768).astype(np.int16)

        try:
            if self.filter_enabled:
                self.process.stdin.write(int_data.tobytes())
                self.process.stdin.flush()

                out_bytes = self.process.stdout.read(frames * 2)
                out_int16 = np.frombuffer(out_bytes, dtype=np.int16)
                out_float32 = out_int16.astype(np.float32) / 32768.0
            else:
                out_float32 = processed  # bypass filter

            # Apply output volume
            outdata[:, 0] = out_float32 * self.output_volume

        except Exception as e:
            print(f"Error: {e}")

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
                    blocksize=480,  # 10ms
                    dtype='float32',
                    callback=self.audio_callback
                )
                self.stream.start()
                self.status_label.setText("Status: Running")
            except Exception as e:
                self.status_label.setText(f"Error: {e}")
                self.process = None
                self.stream = None

    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.process is not None:
            self.process.terminate()
            self.process = None

        self.status_label.setText("Status: Stopped")


if __name__ == "__main__":
    app = QApplication([])
    window = AudioFilterApp()
    window.show()
    app.exec_()
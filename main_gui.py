import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QCheckBox, QLabel, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sounddevice as sd
import torch
from denoiser.demucs import DemucsStreamer
from denoiser.pretrained import get_model, dns48

class AudioProcessor(QThread):
    update_status = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, device="cpu", dry=0.04, num_frames=1):
        super().__init__()
        self.model = dns48().to(device)
        self.model.eval()
        self.streamer = DemucsStreamer(self.model, dry=dry, num_frames=num_frames)
        self.device_in = None
        self.device_out = None
        self.running = False
        self.enable_denoise = False
        self.setup_audio_streams()

    def setup_audio_streams(self):
        try:
            # Input stream (microphone)
            caps = sd.query_devices(None, "input")
            channels_in = min(caps['max_input_channels'], 2)
            self.stream_in = sd.InputStream(
                device=None,
                samplerate=self.model.sample_rate,
                channels=channels_in)

            # Output stream (speakers)
            caps = sd.query_devices(None, "output")
            channels_out = min(caps['max_output_channels'], 2)
            self.stream_out = sd.OutputStream(
                device=None,
                samplerate=self.model.sample_rate,
                channels=channels_out)
        except Exception as e:
            self.error_occurred.emit(f"Audio device error: {str(e)}")
            raise
    
    def query_devices(self, device, kind):
        try:
            caps = sd.query_devices(device, kind=kind)
        except ValueError:
            message = bold(f"Invalid {kind} audio interface {device}.\n")
            message += (
                "If you are on Mac OS X, try installing Soundflower "
                "(https://github.com/mattingalls/Soundflower).\n"
                "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
                "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
                "audio interface to use this.")
            print(message, file=sys.stderr)
            sys.exit(1)
        return caps

    def run(self):
        self.running = True
        device_in = 1
        caps = self.query_devices(device_in, "input")
        channels_in = min(caps['max_input_channels'], 2)
        self.stream_in = sd.InputStream(
            device=device_in,
            samplerate=self.model.sample_rate,
            channels=channels_in)

        device_out = 3
        caps = self.query_devices(device_out, "output")
        channels_out = min(caps['max_output_channels'], 2)
        self.stream_out = sd.OutputStream(
            device=device_out,
            samplerate=self.model.sample_rate,
            channels=channels_out)
        
        self.stream_in.start()
        self.stream_out.start()
        self.update_status.emit("Processing started...")

        first = True
        device = next(self.model.parameters()).device  # Get device from model parameters
        
        while self.running:
            try:
                length = self.streamer.total_length if first else self.streamer.stride
                first = False
                
                # Read audio frame
                frame, overflow = self.stream_in.read(length)
                frame = torch.from_numpy(frame).mean(dim=1).to(device)  # Use the device we got
                
                if self.enable_denoise:
                    # Process with denoiser
                    with torch.no_grad():
                        out = self.streamer.feed(frame[None])[0]
                    if not out.numel():
                        continue
                    out = 0.99 * torch.tanh(out)  # Compressor
                else:
                    # Pass through original audio
                    out = frame
                
                # Prepare output
                out = out[:, None].repeat(1, 2)  # Stereo output
                out.clamp_(-1, 1)
                out = out.cpu().numpy()
                
                # Write output
                self.stream_out.write(out)

            except Exception as e:
                self.error_occurred.emit(f"Processing error: {str(e)}")
                break

    def stop(self):
        self.running = False
        self.stream_out.stop()
        self.stream_in.stop()
        self.update_status.emit("Processing stopped")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Denoiser")
        self.setFixedSize(300, 150)
        
        # Create processor thread
        self.processor = AudioProcessor()
        self.processor.update_status.connect(self.update_status)
        self.processor.error_occurred.connect(self.show_error)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # Checkbox for denoising
        self.denoise_checkbox = QCheckBox("Enable Noise Removal")
        self.denoise_checkbox.stateChanged.connect(self.toggle_denoise)
        
        # Status label
        self.status_label = QLabel("Status: Not running")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layout
        layout.addWidget(self.denoise_checkbox)
        layout.addWidget(self.status_label)
        main_widget.setLayout(layout)
        
        # Start processing thread
        self.processor.start()

    def toggle_denoise(self, state):
        self.processor.enable_denoise = state == Qt.Checked
        if state == Qt.Checked:
            self.status_label.setText("Noise removal: ON")
        else:
            self.status_label.setText("Noise removal: OFF")

    def update_status(self, message):
        self.status_label.setText(message)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.close()

    def closeEvent(self, event):
        self.processor.stop()
        self.processor.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
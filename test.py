import sounddevice as sd
import numpy as np
import noisereduce as nr
from scipy import signal

# Audio parameters
RATE = 44100
CHUNK = 1024
DTYPE = 'float32'

# Initialize noise profile (record some silence/noise first)
noise_sample = np.random.normal(0, 0.01, RATE)  # placeholder

def apply_bandpass(audio_data, lowcut=300, highcut=3000, fs=RATE, order=5):
    """Apply bandpass filter to remove very low and high frequencies"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, audio_data)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    
    audio_data = indata[:, 0]
    
    # Step 1: Noise reduction
    reduced_noise = nr.reduce_noise(
        y=audio_data, 
        sr=RATE,
        y_noise=noise_profile,
        stationary=True
    )
    
    # Step 2: Bandpass filter
    filtered = apply_bandpass(reduced_noise)
    
    # Step 3: Normalize volume
    filtered = filtered / np.max(np.abs(filtered))
    
    # Output the processed audio
    sd.play(filtered, samplerate=RATE)

# Start stream
with sd.InputStream(
    samplerate=RATE,
    blocksize=CHUNK,
    channels=1,
    dtype=DTYPE,
    callback=audio_callback
):
    input("Press Enter to stop...")
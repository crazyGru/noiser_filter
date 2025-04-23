import sounddevice as sd
import numpy as np

# Audio Config
samplerate = 44100 # CD quality
blocksize = 1024

# Simple placeholder filter (bypass or light noise gate)
def noise_filter(indata):
    threshold = 0.01
    filtered = np.where(np.abs(indata) < threshold, 0, indata)
    return filtered

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    
    outdata[:] = noise_filter(indata)

# Start real-time stream
with sd.Stream(channels=1,
               samplerate=samplerate,
               blocksize=blocksize,
               callback=audio_callback):
    print("Running... Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopped.")
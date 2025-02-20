import numpy as np
import librosa

sr=22050

# Load audio file
# Load audio file
filepath = './billy.wav'
y, fs = librosa.load(filepath, sr=sr, duration=20.0)

# Compute STFT
n_fft = 1024  # FFT window size
hop_length = 256  # Hop size
X = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)  # Complex STFT matrix

# Compute magnitude and phase
S = np.abs(X)  # Magnitude spectrogram
phase = np.angle(X)  # Phase spectrogram

# Print matrices
np.set_printoptions(suppress=True, precision=4)  # Format for readability
print("Magnitude Spectrogram (S):")
print(S)

print("\nPhase Spectrogram:")
print(phase)

import librosa
import numpy as np
import soundfile as sf

# Load the audio file
y, sr = librosa.load('billy_recon.wav', sr=16000)

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y)

# Get the magnitude spectrogram
magnitude = np.abs(D)

def griffin_lim(magnitude, n_iter=2, alpha=0.99, lambda_=0.1):
    # Initialize the phase randomly
    phase = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    S = magnitude * phase

    for i in range(n_iter):
        # Inverse STFT
        y = librosa.istft(S)

        # STFT
        S = librosa.stft(y)

        # Update the magnitude
        S = magnitude * np.exp(1j * np.angle(S))

        # Apply alpha and lambda
        S = (1 / (1 + lambda_)) * (S + (lambda_ / (1 + lambda_)) * magnitude * np.exp(1j * np.angle(S)))

    return librosa.istft(S)

# Apply the custom Griffin-Lim algorithm
reconstructed_audio = griffin_lim(magnitude, n_iter=2, alpha=0.5, lambda_=0.5)

# Save the reconstructed audio
sf.write('reconstructed_billy_custom.wav', reconstructed_audio, sr)
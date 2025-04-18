import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

pwrThd = -10


# Load the audio file
# y, sr = librosa.load('./billy-inputs/billy_recon.wav', sr=16000)
y, sr = librosa.load('./billy_recon.wav', sr=16000)

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
reconstructed_audio = griffin_lim(magnitude, n_iter=2, alpha=0.99, lambda_=0.01)

# Save the reconstructed audio and plot
sf.write('reconstructed_billy_custom.wav', reconstructed_audio, sr)
# sf.write('reconstructed_michelle_custom.wav', reconstructed_audio, sr)
plt.figure(figsize=(10,4))
librosa.display.waveshow(reconstructed_audio, sr=sr)
plt.savefig("billy-outputs/iterated-waveform.png", dpi=300)
plt.show()



# spectrogram of GLA waveform
mel = librosa.feature.melspectrogram(  y=reconstructed_audio,
                                        sr=sr, 
                                        n_fft=2048, 
                                        hop_length=512, 
                                        win_length=None, 
                                        window='hann', 
                                        center=True, 
                                        pad_mode='reflect', 
                                        power=1.0,
                                        n_mels=256)
mel_gla_db = librosa.power_to_db(mel, ref=np.max)
mel[mel < pwrThd] = pwrThd
plt.figure(figsize=(10,2))
librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.savefig("billy-outputs/iterated-spectrogram.png", dpi=300)
plt.show()
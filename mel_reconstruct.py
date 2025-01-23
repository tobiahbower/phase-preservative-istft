import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import istft
from PIL import Image

def mel_spectrogram_to_waveform(image_path, output_wav_path, sample_rate=22050):
    # Load the mel spectrogram image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    mel_spectrogram = np.array(img) / 255.0  # Normalize pixel values to [0, 1]

    # Reverse mel scaling to linear frequency scale (simple approximation)
    num_mel_bins, num_frames = mel_spectrogram.shape
    min_freq = 0
    max_freq = sample_rate / 2
    mel_scale = np.linspace(min_freq, max_freq, num_mel_bins)
    linear_spectrogram = np.zeros_like(mel_spectrogram)

    for i, mel_freq in enumerate(mel_scale):
        linear_spectrogram[i, :] = mel_spectrogram[i, :]

    # Approximate the complex spectrogram
    magnitude_spectrogram = linear_spectrogram
    phase = np.random.uniform(0, 2 * np.pi, magnitude_spectrogram.shape)
    complex_spectrogram = magnitude_spectrogram * np.exp(1j * phase)

    # Perform the inverse Short-Time Fourier Transform (iSTFT)
    _, reconstructed_waveform = istft(complex_spectrogram, fs=sample_rate)

    # Normalize waveform to fit in [-1, 1]
    reconstructed_waveform = reconstructed_waveform / np.max(np.abs(reconstructed_waveform))

    # Save as a WAV file
    write(output_wav_path, sample_rate, (reconstructed_waveform * 32767).astype(np.int16))

    # Plot the reconstructed waveform
    plt.figure(figsize=(10, 4))
    plt.plot(reconstructed_waveform, color='blue')
    plt.title('Reconstructed Waveform')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

# Example usage
image_path = 'AnastasisBiblicalGreek.jpg'  # Path to the mel spectrogram image
output_wav_path = 'AnastasisBiblicalGreekRecon.wav'  # Output WAV file path
mel_spectrogram_to_waveform(image_path, output_wav_path)

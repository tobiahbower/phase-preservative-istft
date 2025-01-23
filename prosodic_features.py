import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
import os
from pydub.playback import play

# Set ffmpeg path explicitly
AudioSegment.converter = "C:\\ffmpeg\\bin"

# Function to convert MP3 to WAV (librosa does not support MP3 directly)
def mp3_to_wav(input_file):
    audio = AudioSegment.from_mp3(input_file)
    wav_file = input_file.replace(".mp3", ".wav")
    audio.export(wav_file, format="wav")
    return wav_file

# Function to extract and plot prosodic features
def extract_prosodic_features(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)  # Preserve original sampling rate

    # Extract pitch (F0 contour)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_contour = [
        np.max(pitches[:, i]) if np.max(magnitudes[:, i]) > 0.1 else 0
        for i in range(pitches.shape[1])
    ]

    # Extract energy (amplitude envelope)
    energy = librosa.feature.rms(y=y)[0]

    # Generate time axes
    time_pitch = np.linspace(0, len(y) / sr, num=len(pitch_contour))
    time_energy = librosa.frames_to_time(np.arange(len(energy)), sr=sr)

    # Plot pitch contour
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_pitch, pitch_contour, label="Pitch (F0 Contour)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Contour")
    plt.legend()

    # Plot energy envelope
    plt.subplot(2, 1, 2)
    plt.plot(time_energy, energy, label="Energy Envelope", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.title("Energy Envelope")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    # Input MP3 file
    input_file = input("Enter the path to your MP3 file: ")

    if not input_file.endswith(".mp3"):
        print("Please provide a valid MP3 file.")
    elif not os.path.exists(input_file):
        print("File not found. Please check the file path.")
    else:
        # Convert MP3 to WAV
        wav_file = mp3_to_wav(input_file)

        # Extract and plot prosodic features
        extract_prosodic_features(wav_file)

        # Cleanup: Remove the temporary WAV file
        os.remove(wav_file)

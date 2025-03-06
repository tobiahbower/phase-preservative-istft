import os
import librosa
import librosa.display
import IPython.display as ip
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

plotPSD = True
plotWaveforms = True
plotSpectral = True
pwrThd = -10
sr=22050 # higher frequencies lead to worse GLA performance!

# clear terminal
os.system('cls')
plt.ioff()

# Load audio file
# filepath = './michelle-inputs/michelle.wav'
filepath = './billy-inputs/billy.wav'
y, fs = librosa.load(filepath, sr=sr, duration=20.0)
print("Sample rate:", fs, "Hz\n")

# short time fourier transform
y_stft = np.abs(librosa.stft(y))
y_stft_db = librosa.amplitude_to_db(np.abs(y_stft), ref=np.max)

if plotPSD:
    plt.figure(figsize=(10,4))
    librosa.display.specshow(y_stft_db, x_axis='time', y_axis='log', sr=fs)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Power Spectrogram")
    plt.show(block=False)

# create spectrogram
mel = librosa.feature.melspectrogram(  y=y,
                                        sr=fs, 
                                        n_fft=2048, 
                                        hop_length=512, 
                                        win_length=None, 
                                        window='hann', 
                                        center=True, 
                                        pad_mode='reflect', 
                                        power=1.0,
                                        n_mels=256)
mel_db = librosa.power_to_db(mel, ref=np.max)
mel[mel < pwrThd] = pwrThd

# inverse to waveform from mel
y_recon = librosa.feature.inverse.mel_to_audio( mel, 
                                                sr=fs, 
                                                n_fft=2048, 
                                                hop_length=512, 
                                                win_length=None, 
                                                window='hann', 
                                                center=True, 
                                                pad_mode='reflect', 
                                                power=1.0, 
                                                n_iter=256)


# get the spectral data from the spectrogram
y_stft_mel = librosa.feature.inverse.mel_to_stft(mel)


# inverse using griffin-lim algorithm to estimate phase
y_gla = librosa.griffinlim(y_stft_mel)

# Invert without estimating phase
y_istft = librosa.istft(y_stft_mel)

# write processed waveforms
sf.write('billy_recon.wav', y, fs)
sf.write('billy_recon_gla.wav', y_gla, fs) # decent, but phasing still occurs
sf.write('billy_recon_istft.wav', y_istft, fs) # bad

# diff the waveforms
if plotWaveforms:
    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    librosa.display.waveshow(y, sr=fs, color='b', ax=ax[0])
    ax[0].set(title="Original Waveform", xlabel=None)
    ax[0].label_outer()
    librosa.display.waveshow(y_gla, sr=fs, color='g', ax=ax[1])
    ax[1].set(title='Griffin-Lim Reconstruction', xlabel=None)
    ax[1].label_outer()
    librosa.display.waveshow(y_istft, sr=fs, color='r', ax=ax[2])
    ax[2].set_title('Magnitude-only ISTFT Reconstruction')
    plt.show(block=False)

# spectrogram of GLA waveform
mel_gla = librosa.feature.melspectrogram(  y=y_gla,
                                        sr=fs, 
                                        n_fft=2048, 
                                        hop_length=512, 
                                        win_length=None, 
                                        window='hann', 
                                        center=True, 
                                        pad_mode='reflect', 
                                        power=1.0,
                                        n_mels=256)
mel_gla_db = librosa.power_to_db(mel, ref=np.max)
mel_gla[mel_gla < pwrThd] = pwrThd

# spectrogram of vanilla inverse STFT
mel_istft = librosa.feature.melspectrogram(  y=y_istft,
                                        sr=fs, 
                                        n_fft=2048, 
                                        hop_length=512, 
                                        win_length=None, 
                                        window='hann', 
                                        center=True, 
                                        pad_mode='reflect', 
                                        power=1.0,
                                        n_mels=256)
mel_istft_db = librosa.power_to_db(mel, ref=np.max)
mel_istft[mel_istft < pwrThd] = pwrThd


# diff the spectrograms
if plotSpectral:
    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    i1 = librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=fs, cmap='viridis', ax=ax[0])
    fig.colorbar(i1, ax=[ax[0]], format='%+2.0f dB')
    ax[0].set(title="Mel Spectrogram")
    ax[0].label_outer()
    i2 = librosa.display.specshow(mel_gla, x_axis='time', y_axis='mel', sr=fs, cmap='viridis', ax=ax[1])
    fig.colorbar(i2, ax=[ax[1]], format='%+2.0f dB')
    ax[1].set(title="Griffin-Lim Reconstruction")
    ax[1].label_outer()
    i3 = librosa.display.specshow(mel_istft, x_axis='time', y_axis='mel', sr=fs, cmap='viridis', ax=ax[2])
    fig.colorbar(i3, ax=[ax[2]], format='%+2.0f dB')
    ax[2].set(title="Magnitude-only ISTFT Reconstruction")
    
    plt.show()
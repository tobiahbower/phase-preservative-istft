from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import librosa
import numpy as np

y, fs = librosa.load("./billy_recon_istft.wav")
w1 = np.abs(librosa.stft(y))

_, phase = librosa.magphase(w1)
invphase = -phase
w2 = w1 * np.exp(1j *invphase)

w = w1+w2

sf.write('billy_dephased.wav',w , fs)

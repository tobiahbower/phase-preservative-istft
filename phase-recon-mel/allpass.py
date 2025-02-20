import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

filename = 'billy_recon_gla.wav'
x, fs = sf.read(filename)

def all_pass_filter(inp, c, K):
    delayed_data = np.zeros_like(inp)
    delayed_data[c:] = inp[:-c]
    
    Y = K*delayed_data + inp
    return Y

# delay samples
delay = 5
gain = 1

phase_adjusted_audio = all_pass_filter(x, delay, gain)
sf.write('billy_recon_gla_phase_adjusted.wav', phase_adjusted_audio, fs)
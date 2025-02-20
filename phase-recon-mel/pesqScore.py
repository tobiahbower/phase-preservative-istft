import soundfile as sf
from pesq import pesq
from scipy.io import wavfile
from scipy.signal import resample

sr_pesq_req = 16000 # Hz
x = 'billy_recon.wav'
y = 'reconstructed_billy_custom.wav'


sr, X = wavfile.read(x)
sr, Y = wavfile.read(y)
#ref, sr = sf.read(X)
#deg, sr = sf.read(Y)

Xs = int(len(X) * sr_pesq_req / sr)
Ys = int(len(Y) * sr_pesq_req / sr)

Xr = resample(X, Xs)
Yr = resample(Y, Ys)

score = pesq(sr_pesq_req, Xr, Yr, mode='wb')
print(score)

# A good PESQ score generally ranges from 3.30 to 4.50. Here’s how these scores are typically interpreted:

# 3.30 – 3.79: Attention necessary, but no appreciable effort required. The audio quality is acceptable, but there might be some minor issues that require slight attention.
# 3.80 – 4.50: Complete relaxation possible; no effort required. This range indicates excellent audio quality, where the conversation is clear and effortless for both parties.
# Scores below 3.30 are generally considered poor and can lead to significant frustration and communication difficulties:

# 2.80 – 3.29: Attention necessary; a small amount of effort required. The audio quality is subpar, and listeners may need to ask for repetitions.
# 1.00 – 1.99: No meaning understood with any feasible effort. The audio quality is very poor, making it nearly impossible to understand the conversation.
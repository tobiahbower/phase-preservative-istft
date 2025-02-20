% Sampling rate
sr = 22050;

% Load audio file
filepath = './phase-recon-mel/billy.wav';
[y, fs] = audioread(filepath);

% Resample if necessary
if fs ~= sr
    y = resample(y, sr, fs);
end

% Trim to 20 seconds
y = y(1:min(end, 20 * sr));

% Compute STFT
n_fft = 1024; % FFT window size
hop_length = 256; % Hop size
window = hann(n_fft); % Hann window

% Compute STFT using the spectrogram function
X = stft(y, sr, 'Window', window, 'OverlapLength', hop_length, 'FFTLength', n_fft);

% Compute magnitude and phase
S = abs(X); % Magnitude spectrogram
phase = angle(X); % Phase spectrogram

% Print matrices
disp('Magnitude Spectrogram (S):');
disp(S);

disp('Phase Spectrogram:');
disp(phase);

N=100;
nframes=100;

Xtwz = zeros(N,nframes); % pre-allocate STFT output array
M = length(w);           % M = window length, N = FFT length
zp = zeros(N-M,1);       % zero padding (to be inserted)
xoff = 0;                % current offset in input signal x
Mo2 = (M-1)/2;           % Assume M odd for simplicity here
for m=1:nframes
  xt = x(xoff+1:xoff+M); % extract frame of input data
  xtw = w .* xt;         % apply window to current frame
  xtwz = [xtw(Mo2+1:M); zp; xtw(1:Mo2)]; % windowed, zero padded
  Xtwz(:,m) = fft(xtwz); % STFT for frame m
  xoff = xoff + R;       % advance in-pointer by hop-size R
end
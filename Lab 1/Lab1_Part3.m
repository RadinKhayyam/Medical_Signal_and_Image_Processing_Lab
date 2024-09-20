%% loading the data
clc; close all; clear;
EOG_data = load('EOG_sig.mat');
signal = EOG_data.Sig;
fs = EOG_data.fs;
t = 0 : 1 / fs : (length(signal) - 1) / fs;

%% Q1
clc; close all;

subplot(2,1,1) ;
plot(t,signal(1,:)) ;
title('EOG Signal of Left Eye','Interpreter','latex');
ylabel('Magnitude','interpreter','latex');
xlabel('Time(s)','interpreter','latex');

subplot(2,1,2)
plot(t,signal(2,:)) ;
title('EOG Signal of Right Eye','Interpreter','latex');
ylabel('Magnitude','interpreter','latex');
xlabel('Time(s)','interpreter','latex');
%% Q2
clc; close all;

L = length(signal(1,:)) ;
fft_signal = fft(signal(1,:)) ;
fft_shifted_signal = fftshift(fft_signal)/fs;
f = (-L/2:L/2-1)*(fs/L);

subplot(2,2,1) ;
plot(f,abs(fft_shifted_signal)) ;
title('Fourier transform of left eye signal','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
grid on  ;

L = length(signal(2,:)) ;
fft_signal = fft(signal(2,:)) ;
fft_shifted_signal = fftshift(fft_signal)/fs;
f = (-L/2:L/2-1)*(fs/L);

subplot(2,2,2) ;
plot(f,abs(fft_shifted_signal)) ;
title('Fourier transform of right rye signal','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
grid on  ;

subplot(2,2,3) ;
spectrogram(signal(1,:),hamming(128),64,128,fs,'yaxis');
title('Spectogram of left eye signal','Interpreter','latex');

subplot(2,2,4) ;
spectrogram(signal(2,:),hamming(128),64,128,fs,'yaxis');
title('Spectogram of right eye signal','Interpreter','latex');
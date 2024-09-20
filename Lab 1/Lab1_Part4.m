%% loading the data
clc; close all; clear;
EMG_data = load('EMG_sig.mat');
neuropathym_signal = EMG_data.emg_neuropathym;
myopathym_signal = EMG_data.emg_myopathym;
healthym_signal = EMG_data.emg_healthym;
fs = EMG_data.fs;
t_neuropathym = 0:1/fs:(length(neuropathym_signal) - 1) / fs;
t_myopathym = 0:1/fs:(length(myopathym_signal) - 1) / fs;
t_healthym = 0:1/fs:(length(healthym_signal) - 1) / fs;



%% Q1
clc; close all;

subplot(3,1,1) ;
plot(t_neuropathym,neuropathym_signal) ;
title('EMG signal of neuropathy smaple','Interpreter','latex');
ylabel('Magnitude','interpreter','latex');
xlabel('Time(s)','interpreter','latex');

subplot(3,1,2)
plot(t_myopathym,myopathym_signal) ;
title('EMG signal of myopathy smaple','Interpreter','latex');
ylabel('Magnitude','interpreter','latex');
xlabel('Time(s)','interpreter','latex');

subplot(3,1,3)
plot(t_healthym,healthym_signal) ;
title('EMG signal of healthy smaple','Interpreter','latex');
ylabel('Magnitude','interpreter','latex');
xlabel('Time(s)','interpreter','latex');

%% Q2
clc; close all;

L = length(neuropathym_signal) ;
fft_signal = fft(neuropathym_signal) ;
fft_shifted_signal = fftshift(fft_signal)/fs;
f = (-L/2:L/2-1)*(fs/L);

subplot(2,3,1) ;
plot(f,abs(fft_shifted_signal)) ;
title('Fourier transform of neuropathym signal','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
grid on  ;

L = length(myopathym_signal) ;
fft_signal = fft(myopathym_signal) ;
fft_shifted_signal = fftshift(fft_signal)/fs;
f = (-L/2:L/2-1)*(fs/L);

subplot(2,3,2) ;
plot(f,abs(fft_shifted_signal)) ;
title('Fourier transform of myopathym signal','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
grid on  ;

L = length(healthym_signal) ;
fft_signal = fft(healthym_signal) ;
fft_shifted_signal = fftshift(fft_signal)/fs;
f = (-L/2:L/2-1)*(fs/L);

subplot(2,3,3) ;
plot(f,abs(fft_shifted_signal)) ;
title('Fourier transform of healthym signal','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
grid on  ;

subplot(2,3,4) ;
spectrogram(neuropathym_signal/max(abs(neuropathym_signal)),hamming(1024),512,1024,fs,'yaxis');
title('Spectogram of neuropathym signal','Interpreter','latex');

subplot(2,3,5) ;
spectrogram(myopathym_signal/max(abs(myopathym_signal)),hamming(1024),512,1024,fs,'yaxis');
title('Spectogram of myopathym signal','Interpreter','latex');

subplot(2,3,6) ;
spectrogram(healthym_signal/max(abs(healthym_signal)),hamming(1024),512,1024,fs,'yaxis');
title('Spectogram of healthym signal','Interpreter','latex');


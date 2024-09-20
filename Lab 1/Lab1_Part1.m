% load data
clc;clear; close all;

data=load('EEG_sig.mat');
signal=data.Z;
sampling_frequency=data.des.samplingfreq;
num_samples=data.des.nbsamples;
channel_names=data.des.channelnames;
%% Q1
clc; close all;
t=0:1/sampling_frequency:(num_samples/sampling_frequency);
channel5_signal = signal(5, :);
plot(t(1:end-1),channel5_signal);
title('Channel 5 EEG signal in time domain','Interpreter','latex')
xlabel('t');
ylabel(channel_names(5));
grid on
%% Q2
clc; close all;

time_1=[1,15*sampling_frequency];
time_2=[18*sampling_frequency, 40*sampling_frequency];
time_3=[45*sampling_frequency, 50*sampling_frequency];
time_4=[50*sampling_frequency, length(channel5_signal)];

figure()
subplot(2,2,1)
plot(t(time_1(1):time_1(2)),channel5_signal(time_1(1):time_1(2)));
xlabel('t');
ylabel(channel_names(5));
title('Channel 5 EEG signal for t = 0 : 15s','Interpreter','latex')
grid on


subplot(2,2,2)
plot(t(time_2(1):time_2(2)),channel5_signal(time_2(1):time_2(2)));
xlabel('t');
ylabel(channel_names(5));
title('Channel 5 EEG signal for t = 18 : 40s','Interpreter','latex')
grid on


subplot(2,2,3)
plot(t(time_3(1):time_3(2)),channel5_signal(time_3(1):time_3(2)));
xlabel('t');
ylabel(channel_names(5));
title('Channel 5 EEG signal for t = 45 : 50s','Interpreter','latex')
grid on


subplot(2,2,4)
plot(t(time_4(1):time_4(2)),channel5_signal(time_4(1):time_4(2)));
xlabel('t');
ylabel(channel_names(5));
title('Channel 5 EEG signal for t = 50 : end','Interpreter','latex')
grid on

%% Q3

clc; close all;
t=0:1/sampling_frequency:(num_samples/sampling_frequency);
channel13_signal = signal(13, :);
plot(t(1:end-1),channel13_signal);

hold on
t=0:1/sampling_frequency:(num_samples/sampling_frequency);
channel5_signal = signal(5, :);
plot(t(1:end-1),channel5_signal);

title('Channel 5 and Channel 13 EEG signal in one plot','Interpreter','latex')
xlabel('t');
ylabel("channel 5 and 13 signal");
legend("channel 5","channel 23");
grid on 

%% Q4
clc; close all;

offset = max(max(abs(signal)))/2 ;
disp_eeg(signal,offset,sampling_frequency ,channel_names,'EEG with offset 218.75') ;

%% Q6
clc; close all;

time_1=[2*sampling_frequency, 7*sampling_frequency];
time_2=[30*sampling_frequency, 35*sampling_frequency];
time_3=[42*sampling_frequency, 47*sampling_frequency];
time_4=[50*sampling_frequency, 55*sampling_frequency];


signal1=channel5_signal(time_1(1):time_1(2));
t1=t(time_1(1):time_1(2));

signal2=channel5_signal(time_2(1):time_2(2));
t2=t(time_2(1):time_2(2));

signal3=channel5_signal(time_3(1):time_3(2));
t3=t(time_3(1):time_3(2));

signal4=channel5_signal(time_4(1):time_4(2));
t4=t(time_4(1):time_4(2));

subplot(4,2,1)
plot(t1, signal1);
xlabel('t');
ylabel(channel_names(5));
title('Channel 5 signal for t = 2 : 7s','Interpreter','latex')
grid on

L = length(signal1) ;
F_m = fft(signal1) ;
fm_shift = (-L/2:L/2-1)*(sampling_frequency/L);
F_m_shift = fftshift(F_m)/sampling_frequency;

subplot(4,2,2) ;
hold on;
grid on  ;
title('Fourier transform of first period','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
plot(fm_shift,abs(F_m_shift)) ;

subplot(4,2,3)
plot(t2, signal2);
xlabel('t');
ylabel(channel_names(5));
title('Channel 5 signal for t = 30 : 35s','Interpreter','latex')
grid on

L = length(signal2) ;
F_m = fft(signal2) ;
fm_shift = (-L/2:L/2-1)*(sampling_frequency/L);
F_m_shift = fftshift(F_m)/sampling_frequency;

subplot(4,2,4) ;
hold on;
grid on  ;
title('Fourier transform of second period','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
plot(fm_shift,abs(F_m_shift)) ;

subplot(4,2,5)
plot(t3, signal3);
xlabel('t');
ylabel(channel_names(5));
title('Channel 5 signal for t = 42 : 47s','Interpreter','latex')
grid on

L = length(signal3) ;
F_m = fft(signal3) ;
fm_shift = (-L/2:L/2-1)*(sampling_frequency/L);
F_m_shift = fftshift(F_m)/sampling_frequency;

subplot(4,2,6) ;
hold on;
grid on  ;
title('Fourier transform of third period','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
plot(fm_shift,abs(F_m_shift)) ;


subplot(4,2,7)
plot(t4, signal4);
xlabel('t');
ylabel(channel_names(5));
title('Channel 5 signal for t = 50 : 55s','Interpreter','latex')
grid on

L = length(signal4) ;
F_m = fft(signal4) ;
fm_shift = (-L/2:L/2-1)*(sampling_frequency/L);
F_m_shift = fftshift(F_m)/sampling_frequency;

subplot(4,2,8) ;
hold on;
grid on  ;
title('Fourier transform of fourth period','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
plot(fm_shift,abs(F_m_shift)) ;

%% Q7
clc;

time_1=[2*sampling_frequency, 7*sampling_frequency];
time_2=[30*sampling_frequency, 35*sampling_frequency];
time_3=[42*sampling_frequency, 47*sampling_frequency];
time_4=[50*sampling_frequency, 55*sampling_frequency];


signal1=channel5_signal(time_1(1):time_1(2));
signal2=channel5_signal(time_2(1):time_2(2));
signal3=channel5_signal(time_3(1):time_3(2));
signal4=channel5_signal(time_4(1):time_4(2));

pwelch_signal1 = pwelch(signal1);
pwelch_signal1 = fftshift(pwelch_signal1);
L1 = length(pwelch_signal1);
f1 = sampling_frequency*(-L1/2:L1/2-1)/L1; 

pwelch_signal2 = pwelch(signal2);
pwelch_signal2 = fftshift(pwelch_signal2);
L2 = length(pwelch_signal2);
f2 = sampling_frequency*(-L2/2:L2/2-1)/L2; 

pwelch_signal3 = pwelch(signal3);
pwelch_signal3 = fftshift(pwelch_signal3);
L3 = length(pwelch_signal3);
f3 = sampling_frequency*(-L3/2:L3/2-1)/L3; 

pwelch_signal4 = pwelch(signal4);
pwelch_signal4 = fftshift(pwelch_signal4);
L4 = length(pwelch_signal4);
f4 = sampling_frequency*(-L4/2:L4/2-1)/L4; 

subplot(2,2,1);
plot(f1,pwelch_signal1,'linewidth',1)
xlabel('Frequency (Hz)')
ylabel('PSD (1/Hz)')
xlim([0 120])
title('power spectral density of 2s - 7s','Interpreter','latex');

subplot(2,2,2);
plot(f2,pwelch_signal2,'linewidth',1)
xlabel('Frequency (Hz)')
ylabel('PSD (1/Hz)')
xlim([0 120])
title('power spectral density of 30s - 35s','Interpreter','latex');

subplot(2,2,3);
plot(f3,pwelch_signal3,'linewidth',1)
xlabel('Frequency (Hz)')
ylabel('PSD (1/Hz)')
xlim([0 120])
title('power spectral density of 42s - 47s','Interpreter','latex');

subplot(2,2,4);
plot(f4,pwelch_signal4,'linewidth',1)
xlabel('Frequency (Hz)')
ylabel('PSD (1/Hz)')
xlim([0 120])
title('power spectral density of 50s - 55s','Interpreter','latex');

%% Q8
clc; close all;

time_1=[2*sampling_frequency, 7*sampling_frequency];
time_2=[30*sampling_frequency, 35*sampling_frequency];
time_3=[42*sampling_frequency, 47*sampling_frequency];
time_4=[50*sampling_frequency, 55*sampling_frequency];

signal1=channel5_signal(time_1(1):time_1(2));

signal2=channel5_signal(time_2(1):time_2(2));

signal3=channel5_signal(time_3(1):time_3(2));

signal4=channel5_signal(time_4(1):time_4(2));


subplot(2,2,1);
spectrogram(signal1,hamming(128),64,128,sampling_frequency,'yaxis');
title('Spectogram of 2s - 7s','Interpreter','latex');

subplot(2,2,2);
spectrogram(signal2,hamming(128),64,128,sampling_frequency,'yaxis');
title('Spectogram of 30s - 35s','Interpreter','latex');

subplot(2,2,3);
spectrogram(signal3,hamming(128),64,128,sampling_frequency,'yaxis');
title('Spectogram of 42s - 47s','Interpreter','latex');

subplot(2,2,4);
spectrogram(signal4,hamming(128),64,128,sampling_frequency,'yaxis');
title('Spectogram of 50s - 55s','Interpreter','latex');

%% Q9
clc; close all;

time_2=[30*sampling_frequency, 35*sampling_frequency];

signal2=channel5_signal(time_2(1):time_2(2));

filtered_signal = lowpass(signal2,64,sampling_frequency);
downsampled_signal = downsample(filtered_signal,2);
Fs = sampling_frequency / 2; %new sampling frequency
new_t = 0:1/Fs:((length(downsampled_signal)-1)/Fs);

subplot(3,1,1) ;
plot(new_t,downsampled_signal) ;
title('down sampled signal in time domain','Interpreter','latex');
xlabel('t(s)','interpreter','latex');
grid on  ;

L = length(downsampled_signal) ;
fft_signal = fft(downsampled_signal) ;
fft_shifted_signal = fftshift(fft_signal)/Fs;
f = (-L/2:L/2-1)*(Fs/L);

subplot(3,1,2) ;
plot(f,abs(fft_shifted_signal)) ;
title('Fourier transform of down sampled signal','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
grid on  ;

subplot(3,1,3) ;
spectrogram(downsampled_signal,hamming(128),64,128,Fs,'yaxis');
title('Spectogram of down sampled signal','Interpreter','latex');
grid on  ;



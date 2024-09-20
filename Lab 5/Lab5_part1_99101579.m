%% Lab 5 - Q1
%% Load Data
clc;clear; close all;
ECG_normal = load('normal.mat');
ECG_normal = ECG_normal.normal ;
fs=250;
%% Q1.1
clc;
t = ECG_normal(:,1);
signal = ECG_normal(:,2);


time_nomral = [90,100];
time_noisy = [260,270];

normal_part = time_nomral(1)*fs:time_nomral(2)*fs-1;
noisy_part = time_noisy(1)*fs:time_noisy(2)*fs-1;

normal_signal = signal(normal_part);
noisy_signal = signal(noisy_part);

figure;
subplot(2,2,1);
plot(t(normal_part),normal_signal);
xlim([time_nomral(1), time_nomral(2)]);
ylim([-3 3]);
xlabel('Time(s)','Interpreter','latex');
ylabel('Amplitude(mV)','Interpreter','latex')
title('Normal part of ECG in time domain','Interpreter','latex','FontSize',14);
grid on;

[pxx,f] = pwelch(normal_signal,[],[],[],fs);
subplot(2,2,3);
plot(f,10*log10(pxx));
xlim([0 100]);
xlabel('Frequency(Hz)','Interpreter','latex');
ylabel('PSD(db/Hz)','Interpreter','latex')
title('Normal part of ECG in frequency domain','Interpreter','latex','FontSize',14);
grid on;

subplot(2,2,2);
plot(t(noisy_part),noisy_signal);
xlim([time_noisy(1), time_noisy(2)]);
ylim([-3 3]);
xlabel('Time(s)','Interpreter','latex');
ylabel('Amplitude(mV)','Interpreter','latex')
title('Noisy part of ECG in time domain','Interpreter','latex','FontSize',14);
grid on;

[pxx,f] = pwelch(noisy_signal,[],[],[],fs);
subplot(2,2,4);
plot(f,10*log10(pxx));
xlim([0 100]);
xlabel('Frequency(Hz)','Interpreter','latex');
ylabel('PSD(db/Hz)','Interpreter','latex')
title('Normal part of ECG in frequency domain','Interpreter','latex','FontSize',14);
grid on;

%% Q1.2
clc;

energy_normal_signal = sum(normal_signal.^2);
for f_cutoff=10:60
    filtered_signal = bandpass(normal_signal,[1 f_cutoff],fs);
    energy_filtered_signal = sum(filtered_signal.^2);
    if (energy_filtered_signal >= 0.9*energy_normal_signal)
        break
    end
end

disp(['cut off frequency = ',num2str(f_cutoff)]);

clc;
[noisy_signal_fltr,d1] = bandpass(noisy_signal,[1 f_cutoff],fs);
[normal_signal_fltr,d2] = bandpass(normal_signal,[1 f_cutoff],fs);

% frequency response
freqz(d1,[],fs);
% impulse response
impz(d1,100,fs);

%% Q1.3

figure;
subplot(2,2,1);
plot(t(normal_part),normal_signal);
xlim([time_nomral(1), time_nomral(2)]);
ylim([-3 3]);
xlabel('Time(s)','Interpreter','latex');
ylabel('Amplitude(mV)','Interpreter','latex')
title('Normal signal in time domain','Interpreter','latex','FontSize',14);
grid on;

subplot(2,2,3);
plot(t(normal_part),normal_signal_fltr);
xlim([time_nomral(1), time_nomral(2)]);
ylim([-3 3]);
xlabel('Time(s)','Interpreter','latex');
ylabel('Amplitude(mV)','Interpreter','latex')
title('Normal filtered signal in time domain','Interpreter','latex','FontSize',14);
grid on;

subplot(2,2,2);
plot(t(noisy_part),noisy_signal);
xlim([time_noisy(1), time_noisy(2)]);
ylim([-3 3]);
xlabel('Time(s)','Interpreter','latex');
ylabel('Amplitude(mV)','Interpreter','latex')
title('Noisy signal in time domain','Interpreter','latex','FontSize',14);
grid on;

subplot(2,2,4);
plot(t(normal_part),noisy_signal_fltr);
xlim([time_nomral(1), time_nomral(2)]);
ylim([-3 3]);
xlabel('Time(s)','Interpreter','latex');
ylabel('Amplitude(mV)','Interpreter','latex')
title('Noisy filtered signal in time domain','Interpreter','latex','FontSize',14);
grid on;

%%
clc;
clear;
close all;
load("SSVEP_EEG.mat")
fs = 250;
X = zeros(6, 117917);
% for i = 1:6
%     X(i, :) = bandpass(SSVEP_Signal(i, :), [1 40],fs);
% end
% A = ["Pz", "Oz", "P7", "P8", "O2", "O1"];
% for i = 1:6
%     figure()
%     plot(SSVEP_Signal(i, :))
%     hold on 
%     plot(X(i, :))
%     legend('Original Signal', 'Filtered Signal')
%     title(A(i))
% end
%%
clc
Samples = zeros(6, 15, 5*fs);
channel_1 = zeros(15, 5*fs);
channel_2 = zeros(15, 5*fs);
channel_3 = zeros(15, 5*fs);
channel_4 = zeros(15, 5*fs);
channel_5 = zeros(15, 5*fs);
channel_6 = zeros(15, 5*fs);
for i = 1:15
    start_point = Event_samples(i);
    channel_1(i, :) = X(1, start_point : start_point + (5*fs) -1);
    channel_2(i, :) = X(2, start_point : start_point + (5*fs) -1);
    channel_3(i, :) = X(3, start_point : start_point + (5*fs) -1);
    channel_4(i, :) = X(4, start_point : start_point + (5*fs) -1);
    channel_5(i, :) = X(5, start_point : start_point + (5*fs) -1);
    channel_6(i, :) = X(6, start_point : start_point + (5*fs) -1);
end
%%
clc
figure()

for i = 1:15
    subplot(3, 5, i)
    [pxx, f] = pwelch(channel_1(i, :), [], [], [], fs);
    plot(f, pxx)
    xlabel('frequency')
    ylabel('Pow/freq')
    title(i + "-th trial")
    
    hold on
    grid on
    [pxx, f] = pwelch(channel_2(i, :), [], [], [], fs);
    plot(f, pxx)
    [pxx, f] = pwelch(channel_3(i, :), [], [], [], fs);
    plot(f, pxx)
    [pxx, f] = pwelch(channel_4(i, :), [], [], [], fs);
    plot(f, pxx)
    [pxx, f] = pwelch(channel_5(i, :), [], [], [], fs);
    plot(f, pxx)
    [pxx, f] = pwelch(channel_6(i, :), [], [], [], fs);
    plot(f, pxx)
end
leg = legend("Pz", "Oz", "P7", "P8", "O2", "O1");
leg.Position(1) = 0;
leg.Position(2) = 0.6;

    



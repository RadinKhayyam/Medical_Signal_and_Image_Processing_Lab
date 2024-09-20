%% Lab 4 - Part 1

clc;clear; close all;

ERP_EEG = load('ERP_EEG.mat');
signal = ERP_EEG.ERP_EEG;
signal = transpose(signal);

%% Q1

fs = 240;
t = 0:1/fs:1-1/fs;
counter = 1;
figure;
for N=100:100:2500
    signal_mean = mean(signal(1:N,:));
    subplot(5,5,counter);
    plot(t,signal_mean);
    xlabel('time')
    grid on;
    hold on;
    title(['mean of ',num2str(N),' samples'],'Interpreter','latex');
    counter = counter + 1;
end

%% Q2

clc; close all;

max_values = zeros(1,2550);
mean_values = zeros(2550,240);
for N=1:2550
    mean_values(N,:) = mean(signal(1:N,:));
    signal_max = max(mean_values(N,:));
    max_values(N) = signal_max;
end
plot(max_values,'Linewidth',1);
xlim([1 2550])
title('Max value of first N samples mean','Interpreter','latex');
xlabel('N','Interpreter','latex');
ylabel('max value','Interpreter','latex');

%% Q3

clc; close all;

rms_values = zeros(1,2549);
for i = 1:2549
    rms_values(i) = rms(mean_values(i+1)-mean_values(i));
end
plot(rms_values,'Linewidth',1);
xlabel('i');
grid on;
title('rms value between i and i+1 samples','Interpreter','latex');

%% Q5

clc; close all;

N0 = 900;
mean_1 = mean(signal(1:N0,:));
mean_2 = mean(signal(1:2550,:));
mean_3 = mean(signal(1:(N0/3),:));
mean_4 = mean(signal(randi(2550,1,N0),:));
mean_5 = mean(signal(randi(2550,1,N0/3),:));

figure();
subplot(2,3,1);
plot(mean_1,'Linewidth',1)
xlim([0 240])
title('N = 900','Interpreter','latex');

subplot(2,3,2);
plot(mean_2,'Linewidth',1)
xlim([0 240])
title('N = 2550','Interpreter','latex');

subplot(2,3,3);
plot(mean_3,'Linewidth',1)
xlim([0 240])
title('N = 300','Interpreter','latex');

subplot(2,3,4);
plot(mean_4,'Linewidth',1)
xlim([0 240])
title('N = 900 random samples','Interpreter','latex');

subplot(2,3,5);
plot(mean_5,'Linewidth',1)
xlim([0 240])
title('N = 300 random samples','Interpreter','latex');
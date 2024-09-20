%% loading the data
clc; close all; clear;
ECG_data = load('ECG_sig.mat');
signal = ECG_data.Sig;
fs = ECG_data.sfreq;
t = 0:1/fs:((length(signal)-1))/fs;

%% Q1 - channel one and two signals plot
clc; close all;
subplot(2,1,1)
plot(t,signal(:,1))
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
title('Channel One Signal','Interpreter','latex');
xlim([0 max(t)])

subplot(2,1,2)
plot(t,signal(:,2))
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
title('Channel Two Signal','Interpreter','latex');
xlim([0 max(t)])
%% Q1 - diffrent heart beats in diffrent time in one channel
clc; close all;
time_1=[24*fs, 26*fs];
heartbeat_1 = signal(time_1(1):time_1(2),1);

time_2=[150*fs, 152*fs];
heartbeat_2 = signal(time_2(1):time_2(2),1);

subplot(2,1,1)
plot(t(time_1(1):time_1(2)),heartbeat_1);
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
title('Channel One Signal 24s-26s','Interpreter','latex');

subplot(2,1,2)
plot(t(time_2(1):time_2(2)),heartbeat_2);
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
title('Channel One Signal 150s - 152s','Interpreter','latex');

%% Q1 - diffrent heart beats in diffrent channels in one time
clc; close all;
time_1=[45*fs, 47*fs];
heartbeat_1 = signal(time_1(1):time_1(2),1);
heartbeat_2 = signal(time_1(1):time_1(2),2);

subplot(2,1,1)
plot(t(time_1(1):time_1(2)),heartbeat_1);
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
title('Channel One Signal 45s-47s','Interpreter','latex');

subplot(2,1,2)
plot(t(time_1(1):time_1(2)),heartbeat_2);
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
title('Channel Two Signal 45s-47s','Interpreter','latex');

%% Q1 - PQRST for two diffrent heart beats
clc; close all;
time_1=[round(67.5*fs), round(68.1*fs)];
heartbeat_1 = signal(time_1(1):time_1(2),1);

time_2=[round(23.8*fs), round(24.4*fs)];
heartbeat_2 = signal(time_2(1):time_2(2),1);

subplot(2,1,1)
plot(t(time_1(1):time_1(2)),heartbeat_1);
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
title('Channel One Signal 67.5s-68.1s','Interpreter','latex');
xlim([67.5 68.1])
hold on
x = [67.6306,67.7056,67.7694, 67.8556,67.9667];
y = [-0.19,-0.3,0.785,-0.32,-0.145];
plot(x,y,'O');

subplot(2,1,2)
plot(t(time_2(1):time_2(2)),heartbeat_2);
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
title('Channel One Signal 23.8s - 24.4s','Interpreter','latex');
xlim([23.8 24.4])
hold on
x = [23.9889,24.0167,24.075,24.1833,24.3];
y = [-0.14,-0.195,0.885,-0.29,-0.07];
plot(x,y,'O');

%% Q2 
clc; close all;

R_points = ECG_data.ATRTIMED; 
R_numbers = ECG_data.ANNOTD; 

labelMap = [
        "NOTQRS", "NORMAL", "LBBB", "RBBB", "ABERR", "PVC", "FUSION", "NPC", ...
        "APC", "SVPB", "VESC", "NESC", "PACE", "UNKNOWN", "NOISE", "", "ARFCT", ...
        "", "STCH", "TCH", "SYSTOLE", "DIASTOLE", "NOTE", "MEASURE", "PWAVE", "BBB", ...
        "PACESP", "TWAVE", "RHYTHM", "UWAVE", "LEARN", "FLWAV", "VFON", "VFOFF", ...
        "AESC", "SVESC", "LINK", "NAPC", "PFUS", "WFON", "WFOFF", "RONT"
    ];

R_labels = strings(length(R_numbers),1); 
for i = 1:length(R_points)
    R_labels(i,1) = labelMap(R_numbers(i) + 1); 
end

% because we can't plot all the data in one plot, we just select part of it
time_period = [50, 60]; 
R_points_selected = R_points(R_points >= time_period(1) & R_points<=time_period(2));
R_numbers_selected = R_numbers(R_points >= time_period(1) & R_points<=time_period(2));
R_labels_selected = R_labels(R_points >= time_period(1) & R_points<=time_period(2));

t_selected = t(time_period(1)*fs : (time_period(2))*fs);
samples_selected = time_period(1)*fs : (time_period(2))*fs;


subplot(2,1,1)
plot(t_selected, signal(samples_selected,1));

for i = 1:length(R_points_selected)
    text(R_points_selected(i), 1, R_labels_selected(i), 'FontSize',7, 'HorizontalAlignment', 'center');
end
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
ylim([-1, 1.5])
title('Channel One Signal 50s - 60s','Interpreter','latex');

subplot(2,1,2)
plot(t_selected, signal(samples_selected,2));

for i = 1:length(R_points_selected)
    text(R_points_selected(i), -1.2, R_labels_selected(i), 'FontSize',7, 'HorizontalAlignment', 'center');
end
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
ylim([-1.5, 1])
title('Channel Two Signal 50s - 60s','Interpreter','latex');


%% Q3
clc; close all;

existing_numbers = unique(R_numbers);
label_indices = cell(1, length(existing_numbers));

for i = 1:length(existing_numbers)
    label_indices{i} = find(R_numbers == existing_numbers(i));
end

for i = 1:length(existing_numbers)
    label_idx = label_indices{i};
    if ~isempty(label_idx)
        random_idx = randi(length(label_idx));
        selected_idx = label_idx(random_idx);
        
        % Define the time range for the window
        window_start = round(max((R_points(selected_idx) - 2 )* fs, 1));
        window_end = round(min((R_points(selected_idx) + 2) * fs, length(t)));
        
        % Extract the data for the window
        window_data = signal(window_start:window_end, 1);
        window_time = t(window_start:window_end);
        
        subplot(length(existing_numbers)/2,2,i);
        plot(window_time, window_data,'Linewidth',1);
        hold on; plot(R_points(selected_idx),0,'o','MarkerSize',20);
        title([labelMap(existing_numbers(i) + 1)], 'Interpreter', 'latex');
        xlabel("Time (s)", 'Interpreter', 'latex');
        ylabel("Amplitude (mV)", 'Interpreter', 'latex');
        xlim([t(window_start) t(window_end)])
        
    end
end

%% Q4
clc; close all;

abnormal_beats = find(R_numbers ~= 1);

for i = 600:(length(R_numbers)-360)
    if(ismember(i , abnormal_beats) + ismember(i-1 , abnormal_beats) + ismember(i+1 , abnormal_beats) ==3)
        abnormal_index = i;
        break;
    end
end
abnormal_time = [R_points(abnormal_index -1)-0.5, R_points(abnormal_index+1)+0.5];
normal_time = [54.8 56.8];

t_abnormal=[round(abnormal_time(1)*fs), round(abnormal_time(2)*fs)];
signal_abnoraml = signal(t_abnormal(1):t_abnormal(2),1);

t_normal=[round(normal_time(1)*fs), round(normal_time(2)*fs)];
signal_normal = signal(t_normal(1):t_normal(2),1);

subplot(3,2,1)
plot(t(t_abnormal(1):t_abnormal(2)),signal_abnoraml,'Linewidth',1);
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
title('3 abnormal heart beats','Interpreter','latex');

subplot(3,2,2)
plot(t(t_normal(1):t_normal(2)),signal_normal,'Linewidth',1);
xlabel("Time(s)",'Interpreter','latex')
ylabel("Amplitude(mV)",'Interpreter','latex')
title('3 normal heart beats','Interpreter','latex');

L = length(signal_abnoraml) ;
fft_signal = fft(signal_abnoraml) ;
fft_shifted_signal = fftshift(fft_signal)/fs;
f = (-L/2:L/2-1)*(fs/L);


subplot(3,2,3) ;
plot(f,abs(fft_shifted_signal)) ;
title('Fourier transform of abnormal signal','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
grid on  ;

L = length(signal_normal) ;
fft_signal = fft(signal_normal) ;
fft_shifted_signal = fftshift(fft_signal)/fs;
f = (-L/2:L/2-1)*(fs/L);

subplot(3,2,4) ;
plot(f,abs(fft_shifted_signal)) ;
title('Fourier transform of normal signal','Interpreter','latex');
ylabel('magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');
grid on  ;

subplot(3,2,5) ;
spectrogram(signal_abnoraml,hamming(128),64,128,fs,'yaxis');
title('Spectogram of abnormal signal','Interpreter','latex');


subplot(3,2,6) ;
spectrogram(signal_normal,hamming(128),64,128,fs,'yaxis');
title('Spectogram of normal signal','Interpreter','latex');


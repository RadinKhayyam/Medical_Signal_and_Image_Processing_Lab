%% section a
clc
clear
fs = 250;
L = 100;
%for n_422
load('n_422.mat')
normal_1_n422 = n_422(1:10710);
abnormal_1_n422 = n_422(10711:11210);
normal_2_n422 = n_422(11211:11442);
abnormal_2_n422 = n_422(11442:59710);
abnormal_3_n422 = n_422(61288:75000);

[pnormal_1_n422, ~] = pwelch(normal_1_n422(1:10*fs), hamming(L), 10,100, fs);
[pabnormal_2_n422, ~] = pwelch(abnormal_2_n422(1:10*fs), hamming(L), 10,100, fs);
[pabnormal_3_n422, f] = pwelch(abnormal_3_n422(1:10*fs), hamming(L), 10,100, fs);
ppxs1 = [pnormal_1_n422, pabnormal_2_n422, pabnormal_3_n422];

A = ["normal signal-1-n422", "abnormal signal-2(VT)-n422","abnormal signal-3(VFIB)-n422" ];
figure()
for i = 1:3
    subplot(3, 1, i)
    grid on
    plot(f, (ppxs1(:, i)))
    title(A(i))
    xlabel('frecuency')
    ylabel('P')
end
%log plots
figure()
for i = 1:3
    subplot(3, 1, i)
    grid on
    plot(f, 20*log10(ppxs1(:, i)))
    title(A(i))
    xlabel('frecuency')
    ylabel('20*log_10 P')
end
%% section b
t = 0 : 1/fs : (2499)/fs;
figure()
ecgs = [normal_1_n422(1:10*fs), abnormal_2_n422(1:10*fs), abnormal_3_n422(1:10*fs)];
for i = 1:3
    subplot(3, 1, i)
    grid on
    plot(t, ecgs(:, i))
    title(A(i))
    xlabel('Time(s)')
    ylabel('V')
end

%% section c
T = 10;
OL = 0.5;
step = round(T*fs*(1-OL));
num_of_frames = floor((75000 - (T*fs-step))/step); %59
normal_label_n422 = zeros(1, num_of_frames);


for i = 1:num_of_frames
    start_time = (i-1)*step +1;
    finish_time = (i-1)*step + T*fs;
     if finish_time<=10711   %normal
        normal_label_n422(i)=1 ;
    elseif start_time>10711 &&  finish_time<=11211   %VT
        normal_label_n422(i)=3 ;
    elseif start_time>11211 &&  finish_time<=11442  %normal
        normal_label_n422(i)=1 ;
    elseif start_time>11442 &&  finish_time<=59711 %VT
        normal_label_n422(i)=3 ;
    elseif start_time>59711 &&  finish_time<=61288 %noise
        normal_label_n422(i)=4 ;
    elseif start_time>=61288 %VFIB
        normal_label_n422(i)=2 ;
    else %none
        normal_label_n422(i)=0 ;
     end
end
%% section d
features_n422 = zeros(4, num_of_frames); %bandpower %meanfrequency %medfrequency
for i = 1:num_of_frames
    start_time = (i-1)*step +1;
    finish_time = (i-1)*step + T*fs;
    
    window = n_422(start_time : finish_time);
    features_n422(1, i) = bandpower(window, fs, [0, 62.5]);
    features_n422(2, i) = bandpower(window, fs, [62.5, 125]);
    features_n422(3, i) = meanfreq(window, fs);
    features_n422(4, i) = medfreq(window, fs);
end
%% section e
clc
index_normal = zeros(1, num_of_frames);
index_VFIB = zeros(1, num_of_frames);
index_n=normal_label_n422==1;
for i = 1 : num_of_frames
    if (normal_label_n422(i) == 1)
        index_normal(i) = true;
    end
    if (normal_label_n422(i) == 2)
        index_VFIB(i) = true;
    end
end
index_normal = logical(index_normal);
index_VFIB = logical(index_VFIB);

features_normal_n422 = features_n422(:, index_normal);
features_VFIB_n422 = features_n422(:, index_VFIB);
figure()
TITLES = ["feature one bandpower of first-half frequency", "feature two bandpower of second-half frequency", "feature three meanfreq", "feature four medfreq"];
for i = 1 : 4
    bins = linspace(min(min(features_normal_n422(i, :)), min(features_VFIB_n422(i, :))), max(max(features_normal_n422(i, :)), max(features_VFIB_n422(i, :))), 10);
    subplot(4, 1, i)
    histogram(features_normal_n422(i, :), bins)
    hold on
    histogram(features_VFIB_n422(i, :), bins)
    title(TITLES(i))
    legend('normal', 'VFIB')
end
%% f
ecg_data = n_422;
[alarm_bandpower,t] = va_detect_bandpower(ecg_data,fs);
[alarm_medfreq,t] = va_detect_medfreq(ecg_data,fs);
%% section g
L = 1:59;
index = [L(1, index_normal), L(1, index_VFIB)];
truth = zeros(1, 59);
truth(index_VFIB) = 1;
[c, cm, ind, per] = confusion(truth(1, index), alarm_bandpower(index, 1)');
cm
accuraacy_badnpower = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
sensitivity_bandpower = cm(1,1)/(cm(1,1)+cm(2,1))
specificity_bandpower = cm(2,2)/(cm(2,2)+cm(1,2))

[c,cm,ind,per] = confusion(truth(1,index),alarm_medfreq(index,1)');
cm
accuracy_medfreq = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
sensitivity_medfreq = cm(1,1)/(cm(1,1)+cm(2,1))
specificity_medfreq = cm(2,2)/(cm(2,2)+cm(1,2))


%% section h
new_features_n422 = zeros(6, num_of_frames);
for i = 1: num_of_frames
    start_time = (i-1)*step+1;
    finish_time = (i-1)*step + T*fs;
    window = n_422(start_time : finish_time);
    new_features_n422(1, i) = max(window);
    new_features_n422(2, i) = min(window);
    new_features_n422(3, i) = max(window) - min(window);
    new_features_n422(4, i) = mean(findpeaks(window));
    new_features_n422(5, i) = sum(window == 0);
    new_features_n422(6, i) = var(window);
    
end

%% section I
new_features_normal_n422 = new_features_n422(:, index_normal);
new_features_VFIB_n422 = new_features_n422(:, index_VFIB);
figure()
TITLE = ["feature one: max amp", "feature two: min amp", "feature three: peaktopeak", "feature four: mean peak", "feature five: numbers of cross from zero", "feature six: var"];
for i= 1 : 6
    bins = linspace(min(min(new_features_normal_n422(i, :)), min(new_features_VFIB_n422(i, :))), max(max(new_features_normal_n422(i, :)), max(new_features_VFIB_n422(i, :))), 10);
    subplot(2, 3, i)
    histogram(new_features_normal_n422(i, :), bins)
    hold on
    histogram(new_features_VFIB_n422(i, :), bins)
    legend('normal', 'VFIB')
    title(TITLE(i))
end

%% j
[alarm_meanpeak,t] = va_detect_meanpeak(ecg_data,fs);
[alarm_maxamp,t] = va_detect_maxamp(ecg_data,fs);
%% section k
L = 1:59;
index = [L(1, index_normal), L(1, index_VFIB)];
truth = zeros(1, 59);
truth(index_VFIB) = 1;
[c, cm, ind, per] = confusion(truth(1, index), alarm_meanpeak(index, 1)');
cm
accuraacy_meanpeak = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
sensitivity_meanpeak = cm(1,1)/(cm(1,1)+cm(2,1))
specificity_meanpeak = cm(2,2)/(cm(2,2)+cm(1,2))

[c,cm,ind,per] = confusion(truth(1,index),alarm_maxamp(index,1)');
cm
accuracy_maxamp = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
sensitivity_maxamp = cm(1,1)/(cm(1,1)+cm(2,1))
specificity_maxamp = cm(2,2)/(cm(2,2)+cm(1,2))
 
%% doing all the sections for n_424
% section a_2
clc
close all
load('n_424.mat')
fs = 250;
L = 100;
normal_1_n424 = n_424(1:27248);
abnormal_1_n424 = n_424(24249:53672);
noise_2_n424 = n_424(53673:55133);
abnormal_2_n424 = n_424(55134:58287);
abnormal_3_n424 = n_424(58278:75000);

[pnormal_1_n424, ~] = pwelch(normal_1_n424(1:10*fs), hamming(L), 10,100, fs);
[pabnormal_2_n424, ~] = pwelch(abnormal_2_n424(1:10*fs), hamming(L), 10,100, fs);
[pabnormal_3_n424, f] = pwelch(abnormal_3_n424(1:10*fs), hamming(L), 10,100, fs);
ppxs1 = [pnormal_1_n424, pabnormal_2_n424, pabnormal_3_n424];

A = ["normal signal-1-n424", "abnormal signal-2(VFIB)-n424","abnormal signal-3(NOD)-n424" ];
figure()
for i = 1:3
    subplot(3, 1, i)
    grid on
    plot(f, (ppxs1(:, i)))
    title(A(i))
    xlabel('frecuency')
    ylabel('P')
end
%log plots
figure()
for i = 1:3
    subplot(3, 1, i)
    grid on
    plot(f, 20*log10(ppxs1(:, i)))
    title(A(i))
    xlabel('frecuency')
    ylabel('P')
end
%% section b_2
t = 0 : 1/fs : (2499)/fs;
figure()
ecgs = [normal_1_n424(1:10*fs), abnormal_2_n424(1:10*fs), abnormal_3_n424(1:10*fs)];
for i = 1:3
    subplot(3, 1, i)
    grid on
    plot(t, ecgs(:, i))
    title(A(i))
    xlabel('Time(s)')
    ylabel('V')
end
%% section c_2
T = 10;
OL = 0.5;
step = round(T*fs*(1-OL));
num_of_frames = floor((75000 - (T*fs-step))/step); %59
normal_label_n424 = zeros(1, num_of_frames);


for i = 1:num_of_frames
    start_time = (i-1)*step +1;
    finish_time = (i-1)*step + T*fs;
     if finish_time<=27249   %normal
        normal_label_n424(i)=1 ;
    elseif start_time>27249 &&  finish_time<=53673   %VFIB
        normal_label_n424(i)=2 ;
    elseif start_time>53673 &&  finish_time<=55134  %noise
        normal_label_n424(i)=4 ;
    elseif start_time>55134 &&  finish_time<=58288 %ASYS
        normal_label_n424(i)=3 ;
    elseif start_time>58288 %nod
        normal_label_n424(i)=5 ;
    else %none
        normal_label_n424(i)=0 ;
     end
end

%% section d_2
features_n424 = zeros(4, num_of_frames); %bandpower %meanfrequency %medfrequency
for i = 1:num_of_frames
    start_time = (i-1)*step +1;
    finish_time = (i-1)*step + T*fs;
    
    window = n_424(start_time : finish_time);
    features_n424(1, i) = bandpower(window, fs, [0, 62.5]);
    features_n424(2, i) = bandpower(window, fs, [62.5, 125]);
    features_n424(3, i) = meanfreq(window, fs);
    features_n424(4, i) = medfreq(window, fs);
end
%% section e_2
clc
index_normal = zeros(1, num_of_frames);
index_VFIB = zeros(1, num_of_frames);
index_n=normal_label_n424==1;
for i = 1 : num_of_frames
    if (normal_label_n424(i) == 1)
        index_normal(i) = true;
    end
    if (normal_label_n424(i) == 2)
        index_VFIB(i) = true;
    end
end
index_normal = logical(index_normal);
index_VFIB = logical(index_VFIB);

features_normal_n424 = features_n424(:, index_normal);
features_VFIB_n424 = features_n424(:, index_VFIB);
figure()
TITLES = ["feature one bandpower of first-half frequency", "feature two bandpower of second-hals frequency", "feature three meanfreq", "feature four medfreq"];
for i = 1 : 4
    bins = linspace(min(min(features_normal_n424(i, :)), min(features_VFIB_n424(i, :))), max(max(features_normal_n424(i, :)), max(features_VFIB_n424(i, :))), 10);
    subplot(4, 1, i)
    histogram(features_normal_n424(i, :), bins)
    hold on
    histogram(features_VFIB_n424(i, :), bins)
    title(TITLES(i))
    legend('normal', 'VFIB')
end
%% f_2
ecg_data = n_424;
[alarm_bandpower,t] = va_detect_bandpower_424(ecg_data,fs);
[alarm_meanfreq,t] = va_detect_meanfreq_424(ecg_data,fs);
%% section g_2
L = 1:59;
index = [L(1, index_normal), L(1, index_VFIB)];
truth = zeros(1, 59);
truth(index_VFIB) = 1;
[c, cm, ind, per] = confusion(truth(1, index), alarm_bandpower(index, 1)');
cm
accuraacy_badnpower = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
sensitivity_bandpower = cm(1,1)/(cm(1,1)+cm(2,1))
specificity_bandpower = cm(2,2)/(cm(2,2)+cm(1,2))

[c,cm,ind,per] = confusion(truth(1,index),alarm_meanfreq(index,1)');
cm
accuracy_meanfreq = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
sensitivity_meanfreq = cm(1,1)/(cm(1,1)+cm(2,1))
specificity_meanfreq = cm(2,2)/(cm(2,2)+cm(1,2))
%% section h_2
new_features_n424 = zeros(6, num_of_frames);
for i = 1: num_of_frames
    start_time = (i-1)*step+1;
    finish_time = (i-1)*step + T*fs;
    window = n_424(start_time : finish_time);
    new_features_n424(1, i) = max(window);
    new_features_n424(2, i) = min(window);
    new_features_n424(3, i) = max(window) - min(window);
    new_features_n424(4, i) = mean(findpeaks(window));
    new_features_n424(5, i) = sum(window == 0);
    new_features_n424(6, i) = var(window);
    
end
%% section I_2
new_features_normal_n424 = new_features_n424(:, index_normal);
new_features_VFIB_n424 = new_features_n424(:, index_VFIB);
figure()
TITLE = ["feature one: max amp", "feature two: min amp", "feature three: peaktopeak", "feature four: mean peak", "feature five: numbers of cross from zero", "feature six: var"];
for i= 1 : 6
    bins = linspace(min(min(new_features_normal_n424(i, :)), min(new_features_VFIB_n424(i, :))), max(max(new_features_normal_n424(i, :)), max(new_features_VFIB_n424(i, :))), 10);
    subplot(2, 3, i)
    histogram(new_features_normal_n424(i, :), bins)
    hold on
    histogram(new_features_VFIB_n424(i, :), bins)
    legend('normal', 'VFIB')
    title(TITLE(i))
end
%% J_2
[alarm_zero,t] = va_detect_zero_424(ecg_data,fs);
[alarm_peaktopeak,t] = va_detect_peaktopeak_424(ecg_data,fs);
%% k_2
L = 1:59;
index = [L(1, index_normal), L(1, index_VFIB)];
truth = zeros(1, 59);
truth(index_VFIB) = 1;
[c, cm, ind, per] = confusion(truth(1, index), alarm_zero(index, 1)');
cm
accuraacy_numbersofcrosszero = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
sensitivity_numbersofcrosszero= cm(1,1)/(cm(1,1)+cm(2,1))
specificity_numbersofcrosszero = cm(2,2)/(cm(2,2)+cm(1,2))

[c,cm,ind,per] = confusion(truth(1,index),alarm_peaktopeak(index,1)');
cm
accuracy_peaktopeak = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
sensitivity_peaktopeak = cm(1,1)/(cm(1,1)+cm(2,1))
specificity_peaktopeak = cm(2,2)/(cm(2,2)+cm(1,2))
%% section L
ecg_data = n_422;
alarm_zero_n422 = va_detect_zero_424(ecg_data, fs);
index_normal = zeros(1, num_of_frames);
index_VFIB = zeros(1, num_of_frames);
for i = 1 : num_of_frames
    if (normal_label_n422(i) == 1)
        index_normal(i) = true;
    end
    if (normal_label_n422(i) == 2)
        index_VFIB(i) = true;
    end
end
index_normal = logical(index_normal);
index_VFIB = logical(index_VFIB);
L = 1:59;
index = [L(1, index_normal), L(1, index_VFIB)];
truth = zeros(1, 59);
truth(index_VFIB) =1;
[c, cm, ind, per] = confusion(truth(1, index), alarm_zero_n422(index, 1)');
cm
accuraacy_numbersofcrosszero_for_n422 = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
sensitivity_numbersofcrosszero_for_n422= cm(1,1)/(cm(1,1)+cm(2,1))
specificity_numbersofcrosszero_for_n422 = cm(2,2)/(cm(2,2)+cm(1,2))
%% section L_2
ecg_data = n_424;
[alarm_bandpower_n424, t] = va_detect_bandpower(ecg_data, fs);
index_normal = zeros(1, num_of_frames);
index_VFIB = zeros(1, num_of_frames);
for i = 1 : num_of_frames
    if (normal_label_n424(i) == 1)
        index_normal(i) = true;
    end
    if (normal_label_n424(i) == 2)
        index_VFIB(i) = true;
    end
end
index_normal = logical(index_normal);
index_VFIB = logical(index_VFIB);
L = 1:59;
index = [L(1, index_normal), L(1, index_VFIB)];
truth = zeros(1, 59);
truth(index_VFIB) =1;
[c, cm, ind, per] = confusion(truth(1, index), alarm_bandpower_n424(index, 1)');
cm
accuraacy_badnpower_n422_for_n424 = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
sensitivity_bandpower_n422_for_n424= cm(1,1)/(cm(1,1)+cm(2,1))
specificity__badnpower_n422_for_n424 = cm(2,2)/(cm(2,2)+cm(1,2))
%% section M
load ('n_423.mat')
ecg_data = n_423;
[alarm_bandpower,t] = va_detect_bandpower(ecg_data,fs);
T = 10;
OL = 0.5;
step = round(T*fs*(1-OL));
num_of_frames = floor((75000 - (T*fs-step))/step); %59
normal_label_n426 = zeros(1, num_of_frames);


for i = 1:num_of_frames
    start_time = (i-1)*step +1;
    finish_time = (i-1)*step + T*fs;
     if finish_time<=4057   %normal
        normal_label_n424(i)=1 ;
    elseif start_time>4057 &&  finish_time<=5923   %AFIB
        normal_label_n424(i)=2 ;
    elseif start_time>5923 &&  finish_time<=14942  %normal
        normal_label_n424(i)=1 ;
    elseif start_time>14942 &&  finish_time<= 16019 %AFIB
        normal_label_n424(i)=2 ;
    elseif start_time>16019 %noise
        normal_label_n424(i)=3 ;
    else %none
        normal_label_n424(i)=0 ;
     end
end

L = 1:59;
index = [L(1, index_normal), L(1, index_VFIB)];
truth = zeros(1, 59);
truth(index_VFIB) = 1;
[c, cm, ind, per] = confusion(truth(1, index), alarm_bandpower(index, 1)');
cm
accuraacy_badnpower = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(1,2)+cm(2,1)+cm(2,2))
sensitivity_bandpower = cm(1,1)/(cm(1,1)+cm(2,1))
specificity_bandpower = cm(2,2)/(cm(2,2)+cm(1,2))
%% functions used in f
% all of the functions are listed in this section 

function [alarm,t] = va_detect_bandpower(ecg_data,Fs)
%VA_DETECT  ventricular arrhythmia detection skeleton function
%  [ALARM,T] = VA_DETECT(ECG_DATA,FS) is a skeleton function for ventricular
%  arrhythmia detection, designed to help you get started in implementing your
%  arrhythmia detector.
%
%  This code automatically sets up fixed length data frames, stepping through 
%  the entire ECG waveform with 50% overlap of consecutive frames. You can customize 
%  the frame length  by adjusting the internal 'frame_sec' variable and the overlap by
%  adjusting the 'overlap' variable.
%
%  ECG_DATA is a vector containing the ecg signal, and FS is the sampling rate
%  of ECG_DATA in Hz. The output ALARM is a vector of ones and zeros
%  corresponding to the time frames for which the alarm is active (1) 
%  and inactive (0). T is a vector the same length as ALARM which contains the 
%  time markers which correspond to the end of each analyzed time segment. If Fs 
%  is not entered, the default value of 250 Hz is used. 

  %  Template Last Modified: 3/4/06 by Eric Weiss, 1/25/07 by Julie Greenberg


%  Processing frames: adjust frame length & overlap here
%------------------------------------------------------
frame_sec = 10;  % sec
overlap = 0.5;    % 50% overlap between consecutive frames


% Input argument checking
%------------------------
if nargin < 2
    Fs = 250;  % default sample rate
end
if nargin < 1
    error('You must enter an ECG data vector.');
end
ecg_data = ecg_data(:);  % Make sure that ecg_data is a column vector
 

% Initialize Variables
%---------------------
frame_length = round(frame_sec*Fs);  % length of each data frame (samples)
frame_step = round(frame_length*(1-overlap));  % amount to advance for next data frame
ecg_length = length(ecg_data);  % length of input vector
frame_N = floor((ecg_length-(frame_length-frame_step))/frame_step); % total number of frames
alarm = zeros(frame_N,1);	% initialize output signal to all zeros
t = ([0:frame_N-1]*frame_step+frame_length)/Fs;

% Analysis loop: each iteration processes one frame of data
%----------------------------------------------------------
for i = 1:frame_N
    %  Get the next data segment
    seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
    %  Perform computations on the segment . . .
    feature=bandpower(seg,Fs,[62.5 125]);
    %  Decide whether or not to set alarm . . .
    if feature<2.34
        alarm(i) = 1;
    end
end
end





function [alarm,t] = va_detect_medfreq(ecg_data,Fs)
%VA_DETECT  ventricular arrhythmia detection skeleton function
%  [ALARM,T] = VA_DETECT(ECG_DATA,FS) is a skeleton function for ventricular
%  arrhythmia detection, designed to help you get started in implementing your
%  arrhythmia detector.
%
%  This code automatically sets up fixed length data frames, stepping through 
%  the entire ECG waveform with 50% overlap of consecutive frames. You can customize 
%  the frame length  by adjusting the internal 'frame_sec' variable and the overlap by
%  adjusting the 'overlap' variable.
%
%  ECG_DATA is a vector containing the ecg signal, and FS is the sampling rate
%  of ECG_DATA in Hz. The output ALARM is a vector of ones and zeros
%  corresponding to the time frames for which the alarm is active (1) 
%  and inactive (0). T is a vector the same length as ALARM which contains the 
%  time markers which correspond to the end of each analyzed time segment. If Fs 
%  is not entered, the default value of 250 Hz is used. 

  %  Template Last Modified: 3/4/06 by Eric Weiss, 1/25/07 by Julie Greenberg


%  Processing frames: adjust frame length & overlap here
%------------------------------------------------------
frame_sec = 10;  % sec
overlap = 0.5;    % 50% overlap between consecutive frames


% Input argument checking
%------------------------
if nargin < 2
    Fs = 250;  % default sample rate
end
if nargin < 1
    error('You must enter an ECG data vector.');
end
ecg_data = ecg_data(:);  % Make sure that ecg_data is a column vector
 

% Initialize Variables
%---------------------
frame_length = round(frame_sec*Fs);  % length of each data frame (samples)
frame_step = round(frame_length*(1-overlap));  % amount to advance for next data frame
ecg_length = length(ecg_data);  % length of input vector
frame_N = floor((ecg_length-(frame_length-frame_step))/frame_step); % total number of frames
alarm = zeros(frame_N,1);	% initialize output signal to all zeros
t = ([0:frame_N-1]*frame_step+frame_length)/Fs;

% Analysis loop: each iteration processes one frame of data
%----------------------------------------------------------
for i = 1:frame_N
    %  Get the next data segment
    seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
    %  Perform computations on the segment . . .
    feature=medfreq(seg,Fs);
    %  Decide whether or not to set alarm . . .
    if feature>3
        alarm(i) = 1;
    end
end
end

%% functions used in J
function [alarm,t] = va_detect_maxamp(ecg_data,Fs)
%VA_DETECT  ventricular arrhythmia detection skeleton function
%  [ALARM,T] = VA_DETECT(ECG_DATA,FS) is a skeleton function for ventricular
%  arrhythmia detection, designed to help you get started in implementing your
%  arrhythmia detector.
%
%  This code automatically sets up fixed length data frames, stepping through 
%  the entire ECG waveform with 50% overlap of consecutive frames. You can customize 
%  the frame length  by adjusting the internal 'frame_sec' variable and the overlap by
%  adjusting the 'overlap' variable.
%
%  ECG_DATA is a vector containing the ecg signal, and FS is the sampling rate
%  of ECG_DATA in Hz. The output ALARM is a vector of ones and zeros
%  corresponding to the time frames for which the alarm is active (1) 
%  and inactive (0). T is a vector the same length as ALARM which contains the 
%  time markers which correspond to the end of each analyzed time segment. If Fs 
%  is not entered, the default value of 250 Hz is used. 

  %  Template Last Modified: 3/4/06 by Eric Weiss, 1/25/07 by Julie Greenberg


%  Processing frames: adjust frame length & overlap here
%------------------------------------------------------
frame_sec = 10;  % sec
overlap = 0.5;    % 50% overlap between consecutive frames


% Input argument checking
%------------------------
if nargin < 2
    Fs = 250;  % default sample rate
end
if nargin < 1
    error('You must enter an ECG data vector.');
end
ecg_data = ecg_data(:);  % Make sure that ecg_data is a column vector
 

% Initialize Variables
%---------------------
frame_length = round(frame_sec*Fs);  % length of each data frame (samples)
frame_step = round(frame_length*(1-overlap));  % amount to advance for next data frame
ecg_length = length(ecg_data);  % length of input vector
frame_N = floor((ecg_length-(frame_length-frame_step))/frame_step); % total number of frames
alarm = zeros(frame_N,1);	% initialize output signal to all zeros
t = ([0:frame_N-1]*frame_step+frame_length)/Fs;

% Analysis loop: each iteration processes one frame of data
%----------------------------------------------------------
for i = 1:frame_N
    %  Get the next data segment
    seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
    %  Perform compndutations on the segment . . 
    feature=max(seg);
    %  Decide whether or not to set alarm . . .
    if feature<326
        alarm(i) = 1;
    end
end
end

function [alarm,t] = va_detect_meanpeak(ecg_data,Fs)
%VA_DETECT  ventricular arrhythmia detection skeleton function
%  [ALARM,T] = VA_DETECT(ECG_DATA,FS) is a skeleton function for ventricular
%  arrhythmia detection, designed to help you get started in implementing your
%  arrhythmia detector.
%
%  This code automatically sets up fixed length data frames, stepping through 
%  the entire ECG waveform with 50% overlap of consecutive frames. You can customize 
%  the frame length  by adjusting the internal 'frame_sec' variable and the overlap by
%  adjusting the 'overlap' variable.
%
%  ECG_DATA is a vector containing the ecg signal, and FS is the sampling rate
%  of ECG_DATA in Hz. The output ALARM is a vector of ones and zeros
%  corresponding to the time frames for which the alarm is active (1) 
%  and inactive (0). T is a vector the same length as ALARM which contains the 
%  time markers which correspond to the end of each analyzed time segment. If Fs 
%  is not entered, the default value of 250 Hz is used. 

  %  Template Last Modified: 3/4/06 by Eric Weiss, 1/25/07 by Julie Greenberg


%  Processing frames: adjust frame length & overlap here
%------------------------------------------------------
frame_sec = 10;  % sec
overlap = 0.5;    % 50% overlap between consecutive frames


% Input argument checking
%------------------------
if nargin < 2
    Fs = 250;  % default sample rate
end
if nargin < 1
    error('You must enter an ECG data vector.');
end
ecg_data = ecg_data(:);  % Make sure that ecg_data is a column vector
 

% Initialize Variables
%---------------------
frame_length = round(frame_sec*Fs);  % length of each data frame (samples)
frame_step = round(frame_length*(1-overlap));  % amount to advance for next data frame
ecg_length = length(ecg_data);  % length of input vector
frame_N = floor((ecg_length-(frame_length-frame_step))/frame_step); % total number of frames
alarm = zeros(frame_N,1);	% initialize output signal to all zeros
t = ([0:frame_N-1]*frame_step+frame_length)/Fs;

% Analysis loop: each iteration processes one frame of data
%----------------------------------------------------------
for i = 1:frame_N
    %  Get the next data segment
    seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
    %  Perform compndutations on the segment . . 
    feature=mean(findpeaks(seg));
    %  Decide whether or not to set alarm . . .
    if feature>-30
        alarm(i) = 1;
    end
end
end
%% functions used in f_2
function [alarm,t] = va_detect_bandpower_424(ecg_data,Fs)

frame_sec = 10;  % sec
overlap = 0.5;    % 50% overlap between consecutive frames


% Input argument checking
%------------------------
if nargin < 2
    Fs = 250;  % default sample rate
end
if nargin < 1
    error('You must enter an ECG data vector.');
end
ecg_data = ecg_data(:);  % Make sure that ecg_data is a column vector
 

% Initialize Variables
%---------------------
frame_length = round(frame_sec*Fs);  % length of each data frame (samples)
frame_step = round(frame_length*(1-overlap));  % amount to advance for next data frame
ecg_length = length(ecg_data);  % length of input vector
frame_N = floor((ecg_length-(frame_length-frame_step))/frame_step); % total number of frames
alarm = zeros(frame_N,1);	% initialize output signal to all zeros
t = ([0:frame_N-1]*frame_step+frame_length)/Fs;

% Analysis loop: each iteration processes one frame of data
%----------------------------------------------------------
for i = 1:frame_N
    %  Get the next data segment
    seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
    %  Perform computations on the segment . . 
    feature=bandpower(seg,Fs,[0 40]);

    %  Decide whether or not to set alarm . . .
    if feature < 3168
        alarm(i) = 1;
    end
end
end


function [alarm,t] = va_detect_meanfreq_424(ecg_data,Fs)

frame_sec = 10;  % sec
overlap = 0.5;    % 50% overlap between consecutive frames


% Input argument checking
%------------------------
if nargin < 2
    Fs = 250;  % default sample rate
end
if nargin < 1
    error('You must enter an ECG data vector.');
end
ecg_data = ecg_data(:);  % Make sure that ecg_data is a column vector
 

% Initialize Variables
%---------------------
frame_length = round(frame_sec*Fs);  % length of each data frame (samples)
frame_step = round(frame_length*(1-overlap));  % amount to advance for next data frame
ecg_length = length(ecg_data);  % length of input vector
frame_N = floor((ecg_length-(frame_length-frame_step))/frame_step); % total number of frames
alarm = zeros(frame_N,1);	% initialize output signal to all zeros
t = ([0:frame_N-1]*frame_step+frame_length)/Fs;

% Analysis loop: each iteration processes one frame of data
%----------------------------------------------------------
for i = 1:frame_N
    %  Get the next data segment
    seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
    %  Perform computations on the segment . . 
    feature=meanfreq(seg,Fs);
    %  Decide whether or not to set alarm . . .
    if feature> 0.6
        alarm(i) = 1;
    end
end
end
%% function used in J_2
function [alarm,t] = va_detect_zero_424(ecg_data,Fs)
%
frame_sec = 10;  % sec
overlap = 0.5;    % 50% overlap between consecutive frames


% Input argument checking
%------------------------
if nargin < 2
    Fs = 250;  % default sample rate
end
if nargin < 1
    error('You must enter an ECG data vector.');
end
ecg_data = ecg_data(:);  % Make sure that ecg_data is a column vector
 

% Initialize Variables
%---------------------
frame_length = round(frame_sec*Fs);  % length of each data frame (samples)
frame_step = round(frame_length*(1-overlap));  % amount to advance for next data frame
ecg_length = length(ecg_data);  % length of input vector
frame_N = floor((ecg_length-(frame_length-frame_step))/frame_step); % total number of frames
alarm = zeros(frame_N,1);	% initialize output signal to all zeros
t = ([0:frame_N-1]*frame_step+frame_length)/Fs;

% Analysis loop: each iteration processes one frame of data
%----------------------------------------------------------
for i = 1:frame_N
    %  Get the next data segment
    seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
    %  Perform computations on the segment . . 
    feature=sum(seg==0);
    %  Decide whether or not to set alarm . . .
    if feature > 10
        alarm(i) = 1;
    end
end
end

function [alarm,t] = va_detect_peaktopeak_424(ecg_data,Fs)

frame_sec = 10;  % sec
overlap = 0.5;    % 50% overlap between consecutive frames


% Input argument checking
%------------------------
if nargin < 2
    Fs = 250;  % default sample rate
end
if nargin < 1
    error('You must enter an ECG data vector.');
end
ecg_data = ecg_data(:);  % Make sure that ecg_data is a column vector
 

% Initialize Variables
%---------------------
frame_length = round(frame_sec*Fs);  % length of each data frame (samples)
frame_step = round(frame_length*(1-overlap));  % amount to advance for next data frame
ecg_length = length(ecg_data);  % length of input vector
frame_N = floor((ecg_length-(frame_length-frame_step))/frame_step); % total number of frames
alarm = zeros(frame_N,1);	% initialize output signal to all zeros
t = ([0:frame_N-1]*frame_step+frame_length)/Fs;

% Analysis loop: each iteration processes one frame of data
%----------------------------------------------------------
for i = 1:frame_N
    %  Get the next data segment
    seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
    %  Perform computations on the segment . . 
    feature=max(seg)-min(seg);
    %  Decide whether or not to set alarm . . .
    if feature<317
        alarm(i) = 1;
    end
end
end
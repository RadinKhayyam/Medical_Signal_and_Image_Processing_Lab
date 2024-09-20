% question 3
%% 
clc
clear
load("FiveClass_EEG.mat")
  

% Bandpass filter between 8 - 13
fs = 256; 
N = 4;
Fpass1 = 8;
Fpass2 = 13;
Apass = 1;

h = fdesign.bandpass('N,Fp1,Fp2,Ap', N, Fpass1, Fpass2,Apass, fs);
Hd = design(h, 'cheby1');
Alpha_X = zeros(size(X));
for c=1:30
 Alpha_X(:,c) = filter(Hd,X(:,c));
end

% Bandpass filter between 13 - 30
Fpass1 = 13; 
Fpass2 = 30;
Apass = 1;

h = fdesign.bandpass('N,Fp1,Fp2,Ap', N, Fpass1, Fpass2,Apass, fs);
Hd = design(h, 'cheby1');
Beta_X = zeros(size(X));
for c=1:30
 Beta_X(:,c) = filter(Hd,X(:,c));
end

% Bandpass filter between 4 - 8
Fpass1 = 4; 
Fpass2 = 8;
Apass = 1;

h = fdesign.bandpass('N,Fp1,Fp2,Ap', N, Fpass1, Fpass2,Apass, fs);
Hd = design(h, 'cheby1');
Theta_X = zeros(size(X));
for c=1:30
 Theta_X(:,c) = filter(Hd,X(:,c));
end

% Bandpass filter between 1 - 4
Fpass1 = 1;
Fpass2 = 4; 
Apass = 1; 

h = fdesign.bandpass('N,Fp1,Fp2,Ap', N, Fpass1, Fpass2,Apass, fs);
Hd = design(h, 'cheby1');
Delta_X = zeros(size(X));
for c=1:30
 Delta_X(:,c) = filter(Hd,X(:,c));
end

%plotting for first channel 


figure()
subplot(5,1,1);
plot(0:1/fs:5-1/fs,X(1:5*fs,1));
xlabel('time');
title('Original Signal','Interpreter','latex');

subplot(5,1,2);
plot(0:1/fs:5-1/fs,Delta_X(1:5*fs,1));
xlabel('time');
title('Delta','Interpreter','latex');

subplot(5,1,3);
plot(0:1/fs:5-1/fs,Theta_X(1:5*fs,1));
xlabel('time');
title('Theta','Interpreter','latex');


subplot(5,1,4);
plot(0:1/fs:5-1/fs,Alpha_X(1:5*fs,1));
xlabel('time');
title('Alpha','Interpreter','latex');

subplot(5,1,5);
plot(0:1/fs:5-1/fs,Beta_X(1:5*fs,1));
xlabel('time');
title('Beta','Interpreter','latex');
%%
Alpha_Trials = zeros(fs*10+1,30,200);
for i=1:200
 Alpha_Trials(:,:,i) = Alpha_X(trial(i): trial(i)+256*10,:);
end

Beta_Trials = zeros(fs*10+1,30,200);
for i=1:200
 Beta_Trials(:,:,i) = Beta_X(trial(i): trial(i)+256*10,:);
end


Theta_Trials = zeros(fs*10+1,30,200);
for i=1:200
 Theta_Trials(:,:,i) = Theta_X(trial(i): trial(i)+256*10,:);
end


Delta_Trials = zeros(fs*10+1,30,200);
for i=1:200
 Delta_Trials(:,:,i) = Delta_X(trial(i): trial(i)+256*10,:);
end
%%
Alpha_X_avg = zeros(fs*10+1,30,5);
for i = 1:5
    label_index = find(y == i);
    Alpha_X_avg(:,:,i) = mean(Alpha_Trials(:,:,label_index).^2,3);
end
Beta_X_avg = zeros(fs*10+1,30,5);
for i = 1:5
    label_index = find(y == i);
    Beta_X_avg(:,:,i) = mean(Beta_Trials(:,:,label_index).^2,3);
end
Delta_X_avg = zeros(fs*10+1,30,5);
for i = 1:5
    label_index = find(y == i);
    Delta_X_avg(:,:,i) = mean(Delta_Trials(:,:,label_index).^2,3);
end

Theta_X_avg = zeros(fs*10+1,30,5);
for i = 1:5
    label_index = find(y == i);
    Theta_X_avg(:,:,i) = mean(Theta_Trials(:,:,label_index).^2,3);
end
%%
newWin = ones(1,200)/(sqrt(200));
Alpha_X_avgf = zeros(fs*10+1,30,5);
for i = 1:5
    for j = 1:30
        Alpha_X_avgf(:,j,i) = conv( Alpha_X_avg(:,j,i),newWin,'same');
    end
end


Beta_X_avgf = zeros(fs*10+1,30,5);
for i = 1:5
    for j = 1:30
        Beta_X_avgf(:,j,i) = conv( Beta_X_avg(:,j,i),newWin,'same');
    end
end

Delta_X_avgf = zeros(fs*10+1,30,5);
for i = 1:5
    for j = 1:30
        Delta_X_avgf(:,j,i) = conv( Delta_X_avg(:,j,i),newWin,'same');
    end
end

Theta_X_avgf = zeros(fs*10+1,30,5);
for i = 1:5
    for j = 1:30
        Theta_X_avgf(:,j,i) = conv( Theta_X_avg(:,j,i),newWin,'same');
    end
end
figure()
subplot(2, 2, 1)
plot(0:1/fs:10, Alpha_X_avgf(:, 16, 1))
hold on
plot(0:1/fs:10, Alpha_X_avgf(:, 16, 2))
plot(0:1/fs:10, Alpha_X_avgf(:, 16, 3))
plot(0:1/fs:10, Alpha_X_avgf(:, 16, 4))
plot(0:1/fs:10, Alpha_X_avgf(:, 16, 5))
title('Alpha','Interpreter','latex')
legend('1', '2', '3', '4', '5')
subplot(2, 2, 2)
plot(0:1/fs:10, Beta_X_avgf(:, 16, 1))
hold on
plot(0:1/fs:10, Beta_X_avgf(:, 16, 2))
plot(0:1/fs:10, Beta_X_avgf(:, 16, 3))
plot(0:1/fs:10, Beta_X_avgf(:, 16, 4))
plot(0:1/fs:10, Beta_X_avgf(:, 16, 5))
title('Beta','Interpreter','latex')
legend('1', '2', '3', '4', '5')
subplot(2, 2, 3)
plot(0:1/fs:10, Delta_X_avgf(:, 16, 1))
hold on
plot(0:1/fs:10, Delta_X_avgf(:, 16, 2))
plot(0:1/fs:10, Delta_X_avgf(:, 16, 3))
plot(0:1/fs:10, Delta_X_avgf(:, 16, 4))
plot(0:1/fs:10, Delta_X_avgf(:, 16, 5))
title('Delta', 'Interpreter','latex')
legend('1', '2', '3', '4', '5')
subplot(2, 2, 4)
plot(0:1/fs:10, Theta_X_avgf(:, 16, 1))
hold on
plot(0:1/fs:10, Theta_X_avgf(:, 16, 2))
plot(0:1/fs:10, Theta_X_avgf(:, 16, 3))
plot(0:1/fs:10, Theta_X_avgf(:, 16, 4))
plot(0:1/fs:10, Theta_X_avgf(:, 16, 5))
title('Theta','Interpreter','latex')
legend('1', '2', '3', '4', '5')

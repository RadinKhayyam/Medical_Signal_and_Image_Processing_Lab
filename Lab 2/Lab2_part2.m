%% Q1
clc;clear;
data1=load('NewData1.mat');
data2=load('NewData2.mat');
data3=load('NewData3.mat');
data4=load('NewData4.mat');
fs = 250;

signal_1=data1.EEG_Sig;
signal_2=data2.EEG_Sig;
signal_3=data3.EEG_Sig;
signal_4=data4.EEG_Sig;

plotEEG(signal_1)
title('EEG Signal 1','Fontsize',14,'Interpreter','latex');
plotEEG(signal_2)
title('EEG Signal 2','Fontsize',14,'Interpreter','latex');
plotEEG(signal_3)
title('EEG Signal 3','Fontsize',14,'Interpreter','latex');
plotEEG(signal_4)
title('EEG Signal 4','Fontsize',14,'Interpreter','latex');

%% Q3
% signal 1 and signal 3 are selected
clc;

[F_1,W_1,~] = COM2R(signal_1,32);
components_signal1 = W_1*signal_1;

[F_3,W_3,~] = COM2R(signal_3,32);
components_signal3 = W_3*signal_3;

%% Q4
clc;

Electrodes = load('Electrodes.mat');

plotEEG(components_signal1);
title('Independent Components for signal 1','Fontsize',14,'Interpreter','latex');
plotEEG(components_signal3);
title('Independent Components for signal 3','Fontsize',14,'Interpreter','latex');

figure;
for i=1:21
    subplot(7,3,i)
    [pxx,f] = pwelch(components_signal1(i,:),[],[],[],fs);
    plot(f,pxx,'Linewidth',1);
    title(['PSD of ',num2str(i),'th Source in Signal 1'],'FontSize',14, 'Interpreter','latex');
    xlabel('Frequency (Hz)', 'Interpreter','latex');
    ylabel('PSD (power/Hz)', 'Interpreter','latex');
end

figure;
for i=1:21
    subplot(7,3,i)
    [pxx,f] = pwelch(components_signal3(i,:),[],[],fs);
    plot(f,pxx,'Linewidth',1);
    title(['PSD of ',num2str(i),'th Source in Signal 3'],'FontSize',14, 'Interpreter','latex');
    xlabel('Frequency (Hz)', 'Interpreter','latex');
    ylabel('PSD (power/Hz)', 'Interpreter','latex');
end

figure;
for i=1:21
    subplot(4,6,i)
    plottopomap(Electrodes.Electrodes.X(:,1),Electrodes.Electrodes.Y(:,1),Electrodes.Electrodes.labels(1,:),F_1(:,i))
    title([num2str(i),'th Source in Signal 1'],'FontSize',14, 'Interpreter','latex');
end

figure;
for i=1:21
    subplot(4,6,i)
    plottopomap(Electrodes.Electrodes.X(:,1),Electrodes.Electrodes.Y(:,1),Electrodes.Electrodes.labels(1,:),F_3(:,i))
    title([num2str(i),'th Source in Signal 3'],'FontSize',14, 'Interpreter','latex');
end

%% Q5
clc;
SelSources_1 = [2,3,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21];
SelSources_3 = [2,6,9,10,14,17,18,19,20,21];
signal_1_den = F_1(:,SelSources_1)*components_signal1(SelSources_1,:);
signal_3_den = F_3(:,SelSources_3)*components_signal3(SelSources_3,:);
%% Q6
clc;

plotEEG(signal_1_den);
title('Denoised signal 1','Fontsize',14,'Interpreter','latex');
plotEEG(signal_3_den);
title('Denoised signal 3','Fontsize',14,'Interpreter','latex');
%% Functions
function plotEEG(X) 
load('Electrodes.mat') ;
offset = max(abs(X(:))) ;
feq = 250 ;
ElecName = Electrodes.labels ;
disp_eeg(X,offset,feq,ElecName);
end

function plottopomap(elocsX,elocsY,elabels,data)

% define XY points for interpolation
interp_detail = 100;
interpX = linspace(min(elocsX)-.2,max(elocsX)+.25,interp_detail);
interpY = linspace(min(elocsY),max(elocsY),interp_detail);

% meshgrid is a function that creates 2D grid locations based on 1D inputs
[gridX,gridY] = meshgrid(interpX,interpY);
% Interpolate the data on a 2D grid
interpFunction = TriScatteredInterp(elocsY,elocsX,data);
topodata = interpFunction(gridX,gridY);

% plot map
contourf(interpY,interpX,topodata);
hold on
scatter(elocsY,elocsX,10,'ro','filled');
for i=1:length(elocsX)
    text(elocsY(i),elocsX(i),elabels(i))
end
set(gca,'xtick',[])
set(gca,'ytick',[])
end
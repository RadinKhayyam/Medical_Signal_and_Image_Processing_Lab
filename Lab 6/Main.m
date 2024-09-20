%% Lab 6 - Part a
clear; close all; clc ;

load('ElecPosXYZ') ;

%Forward Matrix=
ModelParams.R = [8 8.5 9.2] ; % Radius of diffetent layers
ModelParams.Sigma = [3.3e-3 8.25e-5 3.3e-3]; 
ModelParams.Lambda = [.5979 .2037 .0237];
ModelParams.Mu = [.6342 .9364 1.0362];

Resolution = 1 ;
[LocMat,GainMat] = ForwardModel_3shell(Resolution, ModelParams) ;

scatter3(LocMat(1,:),LocMat(2,:),LocMat(3,:),'.');
xlabel('x','Fontsize',14,'Interpreter','latex')
ylabel('y','Fontsize',14,'Interpreter','latex')
zlabel('z','Fontsize',14,'Interpreter','latex')
title('Dipoles Locations','FontSize',14,'Interpreter','latex');
%% Part b
clc;
Elec_numbers = 21;
Elec_positions=zeros(3,Elec_numbers);
Elec_names=strings(1,Elec_numbers);

for i=1:Elec_numbers
    Elec_positions(:,i)=ElecPos{1, i}.XYZ;
    Elec_names(1,i)=ElecPos{1, i}.Name;
end

scatter3(LocMat(1,:),LocMat(2,:),LocMat(3,:),'.');
hold on
scatter3(ModelParams.R(3)*Elec_positions(1,:),ModelParams.R(3)*Elec_positions(2,:),ModelParams.R(3)*Elec_positions(3,:),'MarkerEdgeColor','k','MarkerFaceColor','red');
hold on
text(ModelParams.R(3)*Elec_positions(1,:)+0.5,ModelParams.R(3)*Elec_positions(2,:),ModelParams.R(3)*Elec_positions(3,:),Elec_names(1,:),'FontSize',14);
xlabel('x','Fontsize',14,'Interpreter','latex')
ylabel('y','Fontsize',14,'Interpreter','latex')
zlabel('z','Fontsize',14,'Interpreter','latex')
title('Dipoles and Electrodes Locations','FontSize',14,'Interpreter','latex');
%% Part c
clc; 

selected_index = 1299;
location_selected = LocMat(:,selected_index);
normalized_vec_selected = location_selected/sqrt(sum(location_selected.^2));

figure;
scatter3(LocMat(1,:),LocMat(2,:),LocMat(3,:),'.');
hold on
scatter3(ModelParams.R(3)*Elec_positions(1,:),ModelParams.R(3)*Elec_positions(2,:),ModelParams.R(3)*Elec_positions(3,:),'MarkerEdgeColor','k','MarkerFaceColor','red');
hold on
text(ModelParams.R(3)*Elec_positions(1,:)+0.5,ModelParams.R(3)*Elec_positions(2,:),ModelParams.R(3)*Elec_positions(3,:),Elec_names(1,:),'FontSize',14);
hold on
scatter3(LocMat(1,selected_index),LocMat(2,selected_index),LocMat(3,selected_index),'r');
hold on
plot3( [ location_selected(1,1),location_selected(1,1)+normalized_vec_selected(1,1) ] ,[ location_selected(2,1),location_selected(2,1)+normalized_vec_selected(2,1)] , [location_selected(3,1),location_selected(3,1)+normalized_vec_selected(3,1)] ,'Color','black','LineWidth',2)
xlabel('x','Fontsize',14,'Interpreter','latex')
ylabel('y','Fontsize',14,'Interpreter','latex')
zlabel('z','Fontsize',14,'Interpreter','latex')
title('One dipole vector','FontSize',14,'Interpreter','latex');

%% Part d
clc;
Interictal=load('Interictal.mat') ;
interictal_signal=Interictal.Interictal;
signal = interictal_signal(1,:);
gain = GainMat(:,3*(selected_index-1)+1:3*selected_index);
M= gain * normalized_vec_selected * signal;

figure();
for i=1:21
    subplot( 6 ,4 ,i);
    plot( M(i,:));
    xlabel('smaple','Interpreter','latex');
    ylabel('Amplitude','Interpreter','latex');
    title(Elec_names(1,i),'Fontsize',14,'Interpreter','latex')
end

%% Part e
clc;

peaks = cell(2, 21);

for i=1:21
    [pks,locs] = findpeaks(M(i,:),MinPeakDistance=10,SortStr="descend");
    locs(pks < max(pks/2)) = [];
    pks(pks < max(pks/2)) = [];
    peaks{1,i} = pks;
    peaks{2,i} = locs;
end

figure()
for i=1:21
    subplot( 6 ,4 ,i);
    plot( M(i,:));
    hold on;
    plot(peaks{2,i},peaks{1,i},"o",'LineWidth',1);
    xlabel('smaple','Interpreter','latex');
    ylabel('Amplitude','Interpreter','latex');
    title(Elec_names(1,i),'Fontsize',14,'Interpreter','latex')
end

mean_peaks = zeros(21,1);
for i=1:21
    window = (cell2mat(peaks(2,:))-3):(cell2mat(peaks(2,:))+3);
    mean_peaks(i) = mean(abs(M(i,window)));
end

figure()
Display_Potential_3D(ModelParams.R(3),mean_peaks)

%% Part f
clc;
alpha=0.1;
Q_MNE=GainMat.'*inv(GainMat*GainMat.'+alpha*eye(21))*mean_peaks;
Q_MNE_hat=zeros(1,length(Q_MNE)/3);
for i=1:length(Q_MNE)/3
    Q_MNE_hat(1,i)=sqrt(sum(Q_MNE((i-1)*3+1:i*3).^2,'all'));
end

%% Part g
clc;

estimated_index=find(Q_MNE_hat==max(Q_MNE_hat));
location_estimated=LocMat(:,estimated_index);
normalized_vec_estimated=Q_MNE((estimated_index-1)*3+1:estimated_index*3)./sqrt(sum(Q_MNE((estimated_index-1)*3+1:estimated_index*3).^2,'all'));


figure;
scatter3(LocMat(1,:),LocMat(2,:),LocMat(3,:),'.');
hold on
scatter3(ModelParams.R(3)*Elec_positions(1,:),ModelParams.R(3)*Elec_positions(2,:),ModelParams.R(3)*Elec_positions(3,:),'MarkerEdgeColor','k','MarkerFaceColor','red');
hold on
text(ModelParams.R(3)*Elec_positions(1,:)+0.5,ModelParams.R(3)*Elec_positions(2,:),ModelParams.R(3)*Elec_positions(3,:),Elec_names(1,:),'FontSize',14);
hold on
scatter3(LocMat(1,selected_index),LocMat(2,selected_index),LocMat(3,selected_index),'r');
hold on
plot3( [ location_selected(1,1),location_selected(1,1)+normalized_vec_selected(1,1) ] ,[ location_selected(2,1),location_selected(2,1)+normalized_vec_selected(2,1)] , [location_selected(3,1),location_selected(3,1)+normalized_vec_selected(3,1)] ,'Color','black','LineWidth',2)
hold on
scatter3(LocMat(1,estimated_index),LocMat(2,estimated_index),LocMat(3,estimated_index),'r');
hold on
plot3( [ location_estimated(1,1),location_estimated(1,1)+normalized_vec_estimated(1,1) ] ,[ location_estimated(2,1),location_estimated(2,1)+normalized_vec_estimated(2,1)] , [location_estimated(3,1),location_estimated(3,1)+normalized_vec_estimated(3,1)] ,'Color','black','LineWidth',2)
xlabel('x','Fontsize',14,'Interpreter','latex')
ylabel('y','Fontsize',14,'Interpreter','latex')
zlabel('z','Fontsize',14,'Interpreter','latex')
title('One dipole vector','FontSize',14,'Interpreter','latex');

figure;
plot3([0,normalized_vec_estimated(1)],[0,normalized_vec_estimated(2)],[0,normalized_vec_estimated(3)],'LineWidth',2);
hold on;
plot3([0,normalized_vec_selected(1)],[0,normalized_vec_selected(2)],[0,normalized_vec_selected(3)],'LineWidth',2);
legend('Estimated direction','Original direction');
%% Part h
clc;

location_error=sqrt(sum((LocMat(:,selected_index)-location_estimated).^2,'all'));
direction_error = sqrt(sum((normalized_vec_estimated-normalized_vec_selected).^2));

disp(['MSE of location = ',num2str(location_error)]);
disp(['MSE of direction = ',num2str(direction_error)]);
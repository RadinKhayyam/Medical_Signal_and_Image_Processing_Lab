%% Question 1
clc;clear;close all;
image = imread("S3_Q1_utils/thorax_t1.jpg");
image=image(:,:,1);
I = double(image);
imshow(I,[])
x1=89;y1=95;
x2=88;y2=175;
flag=zeros(256);
for i=1:256
    for j=1:256
       if sqrt((i-x1)^2+(j-y1)^2)<30 || sqrt((i-x2)^2+(j-y2)^2)<30
          if I(i,j)<=50 
              flag(i,j)=1;
          end
       end
    end
end
Overlay(I, flag)
%% section 2
x1=140;y1=90;
flag=zeros(256);
for i=1:256
    for j=1:256
       if sqrt((i-x1)^2+(j-y1)^2)<55 
          if I(i,j)<=110 && I(i,j)>=70
              flag(i,j)=1;
          end
       end
    end
end
Overlay(I, flag)
%% Question 2
clc;clear;close all;
pd = imread("S3_Q2_utils/pd.jpg");
pd=pd(:,:,1);
pd=double(pd);
t1 = imread("S3_Q2_utils/t1.jpg");
t1=t1(:,:,1);
t1=double(t1);
t2 = imread("S3_Q2_utils/t2.jpg");
t2=t2(:,:,1);
t2=double(t2);
feature = [reshape(pd,249*213,1) reshape(t1,249*213,1) reshape(t2,249*213,1)];
idx = kmeans(feature,6);
idx_reshape=reshape(idx,249,213);
for i=1:6
    index=(idx_reshape==i);
    J=zeros(size(pd));
    J(index)=1;
    subplot(2,3,i)
    imshow(J,[])
end
%% Question 3
k=6;
centers_ind=[randi(length(feature)) randi(length(feature)) randi(length(feature)) randi(length(feature)) randi(length(feature)) randi(length(feature))];
centers=feature(centers_ind,:);
centers_old=zeros(6,3);
cluster=zeros(53037,1);

while sum(sum(((centers_old-centers).^2)')>0.1)>=1
    centers_old=centers;
    for i=1:53037
        dis=feature(i,:)-centers;
        dis=sqrt(dis(:,1).^2+dis(:,2).^2+dis(:,3).^2 );
        min_index=find(dis==min(dis));
        cluster(i)=min_index(1);
    end
    for j=1:6
        class=cluster==j;
        centers(j,:)=mean(feature(class,:));
    end
end

for i=1:6
    cluster_reshape=reshape(cluster,249,213);
    index=(cluster==i);
    J=zeros(size(pd));
    J(index)=1;
    subplot(2,3,i)
    imshow(J,[])
end
%% Question 4
[centers_fcm,U] = fcm(feature,k);
maxU = max(U);
index1 = find(U(1,:) == maxU);
index2 = find(U(2,:) == maxU);
index3 = find(U(3,:) == maxU);
index4 = find(U(4,:) == maxU);
index5 = find(U(5,:) == maxU);
index6 = find(U(6,:) == maxU);

%plot
subplot(2,3,1)
J=zeros(53037,1);
J(index1,1)=1;
imshow(reshape(J,249,213),[])

subplot(2,3,2)
J=zeros(53037,1);
J(index2,1)=1;
imshow(reshape(J,249,213),[])

subplot(2,3,3)
J=zeros(53037,1);
J(index3,1)=1;
imshow(reshape(J,249,213),[])

subplot(2,3,4)
J=zeros(53037,1);
J(index4,1)=1;
imshow(reshape(J,249,213),[])

subplot(2,3,5)
J=zeros(53037,1);
J(index5,1)=1;
imshow(reshape(J,249,213),[])

subplot(2,3,6)
J=zeros(53037,1);
J(index6,1)=1;
imshow(reshape(J,249,213),[])
%% Function
function Overlay(f, mask)

    m = max(f(:));
    fr = f;
    fg = f + mask / max(mask(:)) * m/2;
    fb = f;
   
    imshow(reshape([fr fg fb],[size(f,1) size(f,2) 3])/m, []);
    
    drawnow;
end
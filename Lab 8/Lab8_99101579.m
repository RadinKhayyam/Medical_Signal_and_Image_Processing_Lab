%% Lab 8 - signal processing lab
%% Q1
clc;clear;close all;
image = imread("S2_Q1_utils/t2.jpg");

original_image = double(image(:,:,1));
noisy_image = original_image+randn(size(original_image))*15;

fft_noisy_img=fftshift(fft2(noisy_image));

size_kernel_1 = 4;
Kernel_1 = zeros(256,256);
coordinate = (256 - size_kernel_1)/2+1:(256 + size_kernel_1)/2;
Kernel_1(coordinate,coordinate) = 1;
Kernel_1 = Kernel_1/sum(Kernel_1,'all');
fft_kernel_1=fft_noisy_img.*Kernel_1;
img_kernel_1=ifft2(ifftshift(fft_kernel_1));

size_kernel_2 = 8;
Kernel_2 = zeros(256,256);
coordinate = (256 - size_kernel_2)/2+1:(256 + size_kernel_2)/2;
Kernel_2(coordinate,coordinate) = 1;
Kernel_2 = Kernel_2/sum(Kernel_2,'all');
fft_kernel_2=fft_noisy_img.*Kernel_2;
img_kernel_2=ifft2(ifftshift(fft_kernel_2));

size_kernel_3 = 16;
Kernel_3 = zeros(256,256);
coordinate = (256 - size_kernel_3)/2+1:(256 + size_kernel_3)/2;
Kernel_3(coordinate,coordinate) = 1;
Kernel_3 = Kernel_3/sum(Kernel_3,'all');
fft_kernel_3=fft_noisy_img.*Kernel_3;
img_kernel_3=ifft2(ifftshift(fft_kernel_3));

size_kernel_4 = 32;
Kernel_4 = zeros(256,256);
coordinate = (256 - size_kernel_4)/2+1:(256 + size_kernel_4)/2;
Kernel_4(coordinate,coordinate) = 1;
Kernel_4 = Kernel_4/sum(Kernel_4,'all');
fft_kernel_4=fft_noisy_img.*Kernel_4;
img_kernel_4=ifft2(ifftshift(fft_kernel_4));


gauss_image = imgaussfilt(original_image,1);

subplot(2,4,2)
imshow(original_image/255)
title('Original image','Interpreter','latex','FontSize',14);
subplot(2,4,3)
imshow(noisy_image/255,[])
title('Noisy image','Interpreter','latex','FontSize',14);
subplot(2,4,5)
imshow(abs(img_kernel_1),[])
title('After Kernel 4*4 filter','Interpreter','latex','FontSize',14);
subplot(2,4,6)
imshow(abs(img_kernel_2),[])
title('After Kernel 8*8 filter','Interpreter','latex','FontSize',14);
subplot(2,4,7)
imshow(abs(img_kernel_3),[])
title('After Kernel 16*16 filter','Interpreter','latex','FontSize',14);
subplot(2,4,8)
imshow(abs(img_kernel_4),[])
title('After Kernel 32*32 filter','Interpreter','latex','FontSize',14);

figure()
subplot(2,2,1)
imshow(abs(original_image) ,[])
title('Original image','Interpreter','latex','FontSize',14);
subplot(2,2,2)
imshow(abs(noisy_image) ,[])
title('Noisy image','Interpreter','latex','FontSize',14);
subplot(2,2,3)
imshow(abs(img_kernel_4) ,[])
title('Previous section','Interpreter','latex','FontSize',14);
subplot(2,2,4)
imshow(abs(gauss_image) ,[])
title('Image gaussfilt','Interpreter','latex','FontSize',14);
%% Q2
clc;clear;close all;
image = imread("S2_Q2_utils/t2.jpg");
original_image = double(image(:,:,1));

sigma = 1.2;
h = Gaussian(sigma, [256 256]);

g=conv2(original_image,h,'same');

G=fftshift(fft2(ifftshift((g))));
H=fftshift(fft2(ifftshift((h))));
F=G./H;
f_hat=fftshift(ifft2(ifftshift(F)));

figure();
subplot(1,3,1);
imshow(original_image/255);
title('Original image','Interpreter','latex','FontSize',14);
subplot(1,3,2);
imshow(g/max(g,[],'all'));
title('Blurred image','Interpreter','latex','FontSize',14);
subplot(1,3,3);
imshow(abs(f_hat)/max(abs(f_hat),[],'all'));
title('Deblurred image','Interpreter','latex','FontSize',14);

g_noise=g+randn(size(g))*0.001;

G_noise=fftshift(fft2(ifftshift((g_noise))));
H=fftshift(fft2(ifftshift((h))));
F=G./H;
f_hat=fftshift(ifft2(ifftshift(F)));

figure();
subplot(1,3,1);
imshow(original_image/255);
title('Original image','Interpreter','latex','FontSize',14);
subplot(1,3,2);
imshow(g_noise/max(g_noise,[],'all'));
title('Blurred image + Noise','Interpreter','latex','FontSize',14);
subplot(1,3,3);
imshow(f_hat/max(f_hat,[],'all'));
title('Deblurred noisy image','Interpreter','latex','FontSize',14);

%% Q3
clc;clear;close all;
image = imread("S2_Q2_utils/t2.jpg");
original_image=image(:,:,1);
resized_image=double(imresize(original_image,[64,64],'bilinear'));

% calculate D
h=[0 1 0;1 2 1;0 1 0];
K=zeros(64,64);
D=zeros(64*64);
K(1:3,1:3)=h;

ind = 1;
for c=1:64
    for r=1:64
        temp=circshift(K,[r-1 c-1]);
        D(ind,:)=reshape(temp,1,64*64);
        ind=ind+1;
    end
end

spy(D);
% now apply D
g=D*reshape(resized_image,64*64,1);
g_reshape=reshape(g,64,64);

g_noise=g_reshape+randn(size(g_reshape))*0.05;
f_hat=pinv(D)*reshape(g_noise,64*64,1);

figure();
subplot(2,2,1);
imshow(resized_image,[]);
title('Orignal image','Interpreter','latex','FontSize',14);
subplot(2,2,2);
imshow(g_reshape,[]);
title('Blured image','Interpreter','latex','FontSize',14);
subplot(2,2,3);
imshow(g_noise,[]);
title('Noisy image','Interpreter','latex','FontSize',14);
subplot(2,2,4);
imshow(reshape(f_hat,64,64),[]);
title('Reconstructed image','Interpreter','latex','FontSize',14);

%% Q4 
clc; close all;

beta=0.01;
f_k=zeros(64*64,1);
updated_value = zeros(1,60);
figure;
for i=1:60
   f_k_plus1=f_k+beta * D'*(g-D*f_k);
   updated_value(i) = abs(sum(f_k_plus1-f_k));
   f_k=f_k_plus1;
   if(mod(i,5)==0)
        subplot(4,3,i/5);
        imshow(reshape(f_k,64,64),[]);
        title(['iteration = ',num2str(i)]);
   end
end

figure;
stem(updated_value);
title('$f_k$ updated value','Interpreter','latex','FontSize',14);
xlabel('iteration','Interpreter','latex');
ylabel('$norm(f_{k+1} - f_k)$','Interpreter','latex');
%% Functions
function g = Gaussian(sigma, dims)

	rows = dims(1);
	cols = dims(2);
    slices = 1;
    D = 2;
    if length(dims)>2
        slices = dims(3);
        D = 3;
    end
    
	% locate centre pixel.
    % For 256x256, centre is at (129,129)
    % For 257x257, centre is still at (129,129)
	cr = ceil( (rows-1)/2 ) + 1;
	cc = ceil( (cols-1)/2 ) + 1;
    cs = ceil( (slices-1)/2) + 1;
    
	% Set the parameter in exponent 
    a = 1 / (2*sigma^2);
	g = zeros(rows,cols,slices);

    for s = 1:slices
        for c = 1:cols
            for r = 1:rows
                r_sh = r - cr;
                c_sh = c - cc;
                s_sh = s - cs;
                g(r,c,s) = exp( -a * (r_sh^2 + c_sh^2 + s_sh^2) );
            end
        end
    end
    
    g = g / (sqrt(2*pi)*sigma)^D;
    
end

%% Question 1

clc ;

img = imread('./S1_Q1_utils/t1.jpg');
slice1 = img(:,:,1);
DFT = fftshift(fft(slice1(128,:),length(slice1(128,:))));

f = -pi:2*pi/length(DFT):pi-1/length(DFT) ;


figure()
subplot(2,1,1);
plot(f,abs(DFT)) ;
title('Magnitude of the fourier transform of the 128th row','Interpreter','latex');
ylabel('Magnitude','interpreter','latex');
xlabel('frequency','interpreter','latex');

subplot(2,1,2);
plot(f,(angle(DFT)));
title('Phase of the fourier transform of the 128th row','Interpreter','latex');
ylabel('Phase','interpreter','latex');
xlabel('frequency','interpreter','latex');


FFT2D = fft2(double(slice1));

figure()
subplot(1,3,1);
imshow(slice1);
title('Main image','interpreter','latex');

subplot(1,3,2);
normalize_coeff = max(max(abs(10*log10(abs((FFT2D)))))) ;
imshow(10*log10(abs((FFT2D)))/normalize_coeff) ;
title('FFT without fftshift','interpreter','latex');

subplot(1,3,3);
normalize_coeff = max(max(abs(10*log10(abs(fftshift((FFT2D))))))) ;
imshow(10*log10(abs(fftshift(FFT2D)))/normalize_coeff)
title('FFT with fftshift','interpreter','latex');

%% Question 2
clc
clear

[X,Y] = ndgrid(-127:128,-127:128);
G = zeros(256);
G(sqrt(X.^2 + Y.^2) < 15) = 1;

F  = zeros(256) ;
F(100,50) = 1 ;
F(120,48) = 2 ;

fft_convolution1 = (fft2(G).*fft2(F));
convolution1 = fftshift(ifft2((fft_convolution1)));
normalize_coeff = max(max(convolution1)) ;

figure()
subplot(1,3,1)
imshow(G);
title('G picture','interpreter','latex')

subplot(1,3,2)
imshow(F);
title('F picture','interpreter','latex')

subplot(1,3,3)
imshow(convolution1/normalize_coeff)
title('convolution of G and F pictures','interpreter','latex')

img = imread('./S1_Q2_utils/pd.jpg') ;
slice1 = img(:,:,1) ;

fft_convolution2 = (fft2(G).*fft2(double(slice1)));
convolution2 = fftshift(ifft2((fft_convolution2)));

convolved_picture = uint8(255*convolution2/max(max(abs(convolution2))));
figure()
subplot(1,2,1)
imshow(slice1)
title('Main image','interpreter','latex')

subplot(1,2,2)
imshow(convolved_picture)
title('Image convolved with G picture','interpreter','latex');

%% Question 3
clear
clc

f = 3; %scale factor
img = imread ('./S1_Q3_utils/ct.jpg');
img = im2double(img);
[M, N, d] = size(img);
DFT = fftshift (fft2 (img));
newM = f*M;
newN = f*N;
newpic = zeros(newM, newN, 3);
Ms = round (M * (f/2 - 0.5));
Ns = round (N * (f/2 - 0.5));
newpic(Ms:(M+Ms-1), Ns:(N+Ns-1), :) = DFT;
final_pic = abs (ifft2 (ifftshift (newpic)));
f_fin = final_pic(Ms:(M+Ms-1), Ns:(N+Ns-1), :);
figure()
subplot(1, 2, 1)
imshow(img)
title('Main image','interpreter','latex')
subplot(1,2,2)
imshow(f_fin * f^2);
title('new zoomed image', 'interpreter','latex');
%% Question 4.1
clear
clc

img = imread('./S1_Q4_utils/ct.jpg');
img = double(img(:,:,1));
slice1 = img(:,:,1) ;


[X,Y] = meshgrid(-127:128,-127:128);

shifted_pic = fftshift(fft2(img)).*exp(-1i*2*pi.*(X*20+Y*40)/256);
shifted = abs(ifft2(ifftshift(shifted_pic)));
normalize_coeff = max(max(shifted)) ;

subplot(1,3,1)
imshow(img/256)
title('Main Image','interpreter','latex');

subplot(1,3,2)
imshow(shifted/normalize_coeff)
title('Shifted Image','interpreter','latex')

subplot(1,3,3)
plot(abs(exp(-1i*2*pi.*(X*20+Y*40)/256)))
title('Magnitude of the kernel','interpreter','latex')

%% Question 4.2
clc
clear

img = imread('./S1_Q4_utils/ct.jpg');
img = img(:,:,1);
rotated_img = imrotate(img,30);

figure()
subplot(1, 2, 1)
imshow(img)
title('Main image','interpreter','latex')
subplot(1, 2, 2)
imshow(rotated_img)
title('Rotated image', 'interpreter', 'latex')
rotated_img1 = rotated_img;
rotated_img = double(rotated_img);
img1 = img;
img = double(img);

DFT_img = fftshift(fft2(img));
DFT_rotated = fftshift(fft2(rotated_img));

Original = 10*log(abs(DFT_img));
Rotated = 10*log(abs(DFT_rotated));

figure()
subplot(2, 1, 1)
plot(Original)
title('fourier of main image',  'interpreter', 'latex')
subplot(2, 1, 2)
plot(Rotated)
title('fourier of rotated image',  'interpreter', 'latex')

DF_image = fft2(ifftshift(img));

X = imrotate(fftshift(DF_image), 30);
rotated_image_usingDFT = abs(fftshift(abs((ifft2(ifftshift((X)))))));
rotated_image_usingDFT = rotated_image_usingDFT/max(max(abs(rotated_image_usingDFT)));

figure()
subplot(1, 3, 1)
imshow(img1)
title('Main image','interpreter','latex')
subplot(1, 3, 2)
imshow(rotated_img1)
title('Rotated image', 'interpreter', 'latex')

subplot(1, 3, 3)
imshow(rotated_image_usingDFT)
title('Rotated image in Fourier Domain', 'interpreter', 'latex');
%% Question 5
clc
clear
img = imread('./S1_Q5_utils/t1.jpg') ;
img1 = img;
img = double(img(:,:,1)) ;
xdirect_grad=(circshift(img,[0,1])-circshift(img,[0,-1]))/2;
xnormalize_coeff  = (max(max(xdirect_grad))-min(min(xdirect_grad)));
xdirect_grad = (xdirect_grad - min(min(xdirect_grad)))/xnormalize_coeff;
ydirect_grad=(circshift(img,[1,0])-circshift(img,[-1,0]))/2;
ynormalize_coeff  = (max(max(ydirect_grad))-min(min(ydirect_grad)));
ydirect_grad = (ydirect_grad - min(min(ydirect_grad)))/ynormalize_coeff;


abs_grad = sqrt((xdirect_grad.^2+ydirect_grad.^2));
absnormalize_coeff = max(max(abs_grad));

abs_grad = abs_grad/absnormalize_coeff;


figure()
subplot(1, 4, 1);
imshow(img1)
title('Main image','interpreter','latex')

subplot(1, 4,2);
imshow(xdirect_grad)
title('x-gradient', 'interpreter','latex');

subplot(1, 4, 3);
imshow(ydirect_grad)
title('y-gradient', 'interpreter','latex');

subplot(1, 4, 4);
imshow(abs_grad)
title('abs gradient', 'interpreter','latex');

%% Question 6

img = imread("./S1_Q1_utils/t1.jpg");

sobel = edge(rgb2gray(img),'Sobel');
canny = edge(rgb2gray(img),'Canny');

figure()
subplot(1,2,1)
imshow(sobel)
title('Edge finding with Sobel','interpreter','latex')
subplot(1,2,2)
imshow(canny)
title('Edge finding with Canny','interpreter','latex')


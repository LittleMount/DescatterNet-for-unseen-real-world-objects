% prepear the data for experiment
% read image -> resize -> expand black band -> save image

clc,clear
close all

% initial
path = './ILSVRC2012_img_val/';
dirname = './exp data(20220907)/';
% rmdir(dirname);
mkdir(dirname);
n = 1;

%% 数据集1 
for i = 1:0
    img = imread([path,num2str(i),'.jpeg']);
    if ndims(img)==3
        img = rgb2gray(img);
    end
    judge_num = std2(img);
    if judge_num<50 % 清洗低对比度图像
        continue
    end
    img = imresize(img,[1400,1400],'bicubic');
    img_exp = padarray(img, [2 236]); 
    str_name = num2str(n+10000);
    imwrite(img_exp,[dirname,str_name(2:end),'.png']);
    if mod(i,10) == 0
        disp(i);
    end
    n = n+1;
end
n = 669;
%% 数据集2
path = './DIV2K_train_HR/';
for i = 1:0%800
    str1 = num2str(i+10000);
    img = imread([path,str1(2:end),'.png']);
    if ndims(img)==3
        img = rgb2gray(img);
    end
    judge_num = std2(img);
    if judge_num<50 % 清洗低对比度图像
        continue
    end
    img = imresize(img,[1400,1400],'bicubic');
    img_exp = padarray(img, [2 236]); 
%     img = imresize(img,[576,576],'bicubic');
%     img_exp = padarray(img, [414 648], 100); 
    str_name = num2str(n+10000);
    imwrite(img_exp,[dirname,str_name(2:end),'.png']);
    if mod(i,10) == 0
        disp(i);
    end
    n = n+1;
end

%% 数据集3
path = './DIV2K_valid_HR/';
for i = 801:800%900
    str1 = num2str(i+10000);
    img = imread([path,str1(2:end),'.png']);
    if ndims(img)==3
        img = rgb2gray(img);
    end
    judge_num = std2(img);
    if judge_num<50 % 清洗低对比度图像
        continue
    end
    img = imresize(img,[1400,1400],'bicubic');
    img_exp = padarray(img, [2 236]); 
%     img = imresize(img,[576,576],'bicubic');
%     img_exp = padarray(img, [414 648], 100); 
    str_name = num2str(n+10000);
    imwrite(img_exp,[dirname,str_name(2:end),'.png']);
    if mod(i,10) == 0
        disp(i);
    end
    n = n+1;
end
% n = 1349;

%% MNIST数据集
n=1350;

%% 数据集4 标准测试图
path = './standard image/';
for i = 1:10
    img = imread([path,num2str(i),'.png']);
    if ndims(img)==3
        img = rgb2gray(img);
    end
    judge_num = std2(img);
    if judge_num<50 % 清洗低对比度图像
%         continue
    end
    img = imresize(img,[1400,1400],'bicubic');
    img_exp = padarray(img, [2 236]); 
%     img = imresize(img,[576,576],'bicubic');
%     img_exp = padarray(img, [414 648], 100); 
    str_name = num2str(n+10000);
    imwrite(img_exp,[dirname,str_name(2:end),'.png']);
    if mod(i,10) == 0
        disp(i);
    end
    n = n+1;
end

%% 全黑、全白的图像
img0 = zeros(size(img_exp));
str_name = num2str(n+10000);
imwrite(img0,[dirname,str_name(2:end),'.png']);
n = n+1;

img1 = ones(size(img_exp));
str_name = num2str(n+10000);
imwrite(img1,[dirname,str_name(2:end),'.png']);
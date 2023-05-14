% 数据预处理
% 流程：读取图片 -> 裁剪 -> 归一化拉伸 -> 保存图片；
clc,clear
close all

% 文件名的批量读取
nongdu = {'0','0.6','1.2','1.8','2.4','2.8','3.2','3.6'};
for i = 1:8
    consentration = nongdu{i};
    path = ['../20221102/',consentration, 'ml/'];    % 指明哪个文件夹
    file_set = dir(fullfile(path,'*.png'));	% 读取后缀为.jpg的文件信息，保存为结构体数组
    name_set = {file_set.name};             % 获取批量的文件名，保存为元胞数组
    save_path = ['./retinex/',consentration,'ml/'];
    mkdir(save_path);

    % 文件名批量获取已完成，下面是遍历使用文件的示例
    for j = 1351:1366
        filename = [path,num2str(j),'.png'];  % 注意组合“文件夹+文件名”才可读取到图片
        img = im2double(imread(filename));         % 执行后续操作
        img_crop = img(528:1455,555:1482);
%         img_crop_r = imresize(img_crop,[256 256]);
%         img_norm = mat2gray(img_crop_r);
        img_retinex = retinex(img_crop);

        imwrite(img_retinex,[save_path,num2str(j),'.png']);
    %     imshow(img);
    %     pause(0.5);     % 暂停0.5s，把图片显示出来
    end

end



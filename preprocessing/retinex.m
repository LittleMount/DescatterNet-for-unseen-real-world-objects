function I0 = retinex(img)
%RETINEX 此处显示有关此函数的摘要
%   此处显示详细说明

sigma=250;	% 高斯滤波器的标准差，可适当调整

if size(img,3)>1	% 处理彩色图像
	for k=1:3
	R=img(:,:,k);
	[N1,M1]=size(R);
	F=zeros(N1,M1);%定义高斯滤波函数
	for i=1:N1
	    for j=1:M1
	   F(i,j)=exp(-((i-N1/2)^2+(j-M1/2)^2)/(2*sigma*sigma));
	    end
	end
	F = F./(sum(F(:)));
	R0=double(R);%R信道
	Rlog=log(R0+1);%取对数
	Rfft2=fft2(R0);%傅立叶二维
	Ffft2=fft2(double(F));%高斯滤波二维傅立叶
	DR0=Rfft2.*Ffft2;%频率域高斯滤波
	DR=ifft2(DR0);%滤波后空间对数域图像
	DRdouble=double(DR);
	DRlog=log(DRdouble+1);%相减得到高频部分
	Rr=Rlog-DRlog;
	EXPRr=exp(Rr);
	MIN = min(min(EXPRr));
	MAX = max(max(EXPRr));
	EXPRr = (EXPRr-MIN)/(MAX-MIN);
	EXPRr=adapthisteq(EXPRr);
	I0(:,:,k)=EXPRr;
	end
else	% 处理灰度图像
	R=img;
	[N1,M1]=size(R);
	F=zeros(N1,M1);%定义高斯滤波函数
	for i=1:N1
	    for j=1:M1
	   F(i,j)=exp(-((i-N1/2)^2+(j-M1/2)^2)/(2*sigma*sigma));
	    end
	end
	F = F./(sum(F(:)));
	R0=double(R);%R信道
	Rlog=log(R0+1);%取对数
	Rfft2=fft2(R0);%傅立叶二维
	Ffft2=fft2(double(F));%高斯滤波二维傅立叶
	DR0=Rfft2.*Ffft2;%频率域高斯滤波
	DR=ifft2(DR0);%滤波后空间对数域图像
	DRdouble=double(DR);
	DRlog=log(DRdouble+1);%相减得到高频部分
	Rr=Rlog-DRlog;
	EXPRr=exp(Rr);
	MIN = min(min(EXPRr));
	MAX = max(max(EXPRr));
	EXPRr = (EXPRr-MIN)/(MAX-MIN);
	EXPRr=imadjust(adapthisteq(EXPRr));
	I0=EXPRr;
end
%%%
% subplot(121),imshow(img), title('原图');
% subplot(122),imshow(I0), title('Retinex');

end


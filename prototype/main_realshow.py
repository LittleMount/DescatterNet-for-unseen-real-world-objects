# 流程：读取图像 -> 图像预处理 -> 输入网络 -> 展示结果
# 图像预处理：图像裁剪、Retinex处理和CLAHE处理；
# 以上过程是循环进行的；
# 需求：测试CPU的恢复速度和GPU的恢复速度

import pco
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet_use_in_hardware.func import *
# from unet_use_in_hardware.tensorflow_unet import *

## 初始化
gap = (np.ones([448, 20], dtype=np.float32)*65535).astype(np.uint16)
cam = pco.Camera()  # 指定相机
exp_time = 0.003
cam.configuration = {'exposure time': exp_time, 'roi': (95, 112, 1886, 1903)}  # (543, 560, 1438, 1455)
cam.record(4, mode='ring buffer')
cam.wait_for_first_image()

# 指定用GPU复原
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载网络
unet = UNet(1, 1)  # 定义UNet网络
unet.load_state_dict(torch.load('unet_2.pkl'))  # 加载训练好的网络参数
unet.to(device)

## 循环模块: 读取图像 -> 预处理 -> 生成input -> 复原结果
n = 120
while True:
    n += 1
    start = time.time()
    # 读取图像
    img, meta = cam.image(image_index=-1)

    '''# 自适应曝光
    if np.max(img[543:1438:2, 560:1455:2]) == 65535:    # 过曝了，缩小曝光时间
        exp_time = exp_time*55000/65535
        cam.exposure_time = exp_time
        print('exp_time: ', exp_time)
    elif np.max(img[543:1438:2, 560:1455:2]) > 50000:   # 最大像素值>50000, 无需调节
        pass
    else:                                               # 最大像素值<50000, 增加曝光时间
        exp_time = exp_time*55000/np.max(img[543:1438:2, 560:1455:2])
        cam.exposure_time = exp_time
        print('exp_time: ', exp_time)'''

    # 数据预处理
    # start = time.time()
    img_pre = pre_process(img)
    # print('图像预处理时间：{:.6f}s'.format(time.time()-start))

    # 生成网络输入所需要的数据
    input = torch.tensor(normalize(img_pre)).reshape([1, 1, 448, 448]).to(device)

    # 用网络复原
    # start = time.time()
    result = unet(input)
    # result1 = unet1(input)
    # print('图像复原时间：{:.6f}s'.format(time.time() - start))
    # result1 = convert2img(result)     img_pre, #
    cv2.imshow('image', np.concatenate((img[::4, ::4], gap, im2uint16(normalize(img[::4, ::4])),
                                        gap, im2uint16(convert2img(result))), axis=1))
                                        # gap, im2uint16(convert2img(result1))), axis=1))
    # plt.figure(), plt.imshow(result1, cmap='gray'), plt.show()

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) == ord('q'):
        break

    # if n > 200:
    #     break

    print(time.time()-start)

cv2.waitKey(-1)
cam.close()
cv2.destroyAllWindows()

''' else
# 绘制图窗
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.namedWindow('img_crop', cv2.WINDOW_NORMAL)
cv2.namedWindow('img_retinex', cv2.WINDOW_NORMAL)
cv2.namedWindow('img_clahe', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 800, 800)
cv2.resizeWindow('img_crop', 800, 800)
cv2.resizeWindow('img_retinex', 800, 800)
cv2.resizeWindow('img_clahe', 800, 800)


plt.figure(1), plt.imshow(img_crop)
plt.axis('off')
plt.show()
runfile('./main_realshow.py')

# cv2.imshow('img', img), cv2.waitKey(1)
# plt.figure(1), plt.hist(img.ravel(), 655, [0, 65535])

# 裁剪数据
img_crop = img[544:1439:2, 561:1456:2]
# cv2.imshow('img_crop', img_crop), cv2.waitKey(1)
# plt.figure(2), plt.hist(img_crop.ravel(), 655, [0, 65535])

# 计算Retinex图像
retinex1 = retinex(img_crop)
img_retinex = np.uint16(retinex1 * 65535)
# cv2.imshow('img_retinex', img_retinex), cv2.waitKey(1)
# plt.figure(3), plt.hist(img_retinex.ravel(), 655, [0, 65535])

# 使用CLAHE处理
clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_retinex)
# cv2.imshow('img_clahe', img_clahe), cv2.waitKey(1)
# plt.figure(4), plt.hist(img_clahe.ravel(), 655, [0, 65535])

# 使用imadjust
img_adjust = py_imadjust(img_clahe)

'''

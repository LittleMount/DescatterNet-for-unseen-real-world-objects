import os
import time
import numpy as np
import scipy.io as scio
import tensorflow as tf
import matplotlib.pyplot as plt

f = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class U_Net():
    def __init__(self):
        # setting the size of image
        self.height = 448
        self.width = 448
        self.channels = 1
        self.shape = (self.height, self.width, self.channels)

        # optimization
        optimizer = Adam(0.008, 0.5)

        # Unet
        self.unet = self.build_unet()
        self.unet.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=[self.metric_fun])
        self.unet.summary()

    def build_unet(self, n_filters=32, dropout=0.1,
                   batchnorm=True, padding='same'):

        # define a conv block
        def conv2d_block(input_tensor, n_filters=16, kernel_size=3,
                         batchnorm=True, padding='same'):
            # the first layer
            x = Conv2D(n_filters, kernel_size,
                       padding=padding)(input_tensor)
            if batchnorm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # the second layer
            x = Conv2D(n_filters, kernel_size,
                       padding=padding)(x)
            if batchnorm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)

            return x

        # construct an input
        img = Input(shape=self.shape)

        # contracting path
        c1 = conv2d_block(img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout * 0.5)(p1)

        c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm, padding=padding)

        # extending path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)

        output = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        return Model(img, output)

    def metric_fun(self, y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=1)

    def train(self, epochs=1001, batch_size=8):
        os.makedirs('./weights', exist_ok=True)
        os.makedirs('./evaluation', exist_ok=True)
        # obtain data
        data_input = scio.loadmat('./mat_data/input_retinex.mat')
        data_label = scio.loadmat('./mat_data/label_retinex.mat')

        # load the trained model
        # self.unet.load_weights(r"./best_model.h5")

        # setting the check point
        callbacks = [EarlyStopping(patience=1000, verbose=2),
                     ReduceLROnPlateau(factor=0.5, patience=50, min_lr=0.00005),
                     ModelCheckpoint('./weights/best_model.h5', verbose=2, save_best_only=True)]

        # training
        results = self.unet.fit(np.expand_dims(data_input['input'], axis=3),
                                np.expand_dims(data_label['label'], axis=3),
                                batch_size=batch_size, epochs=epochs, verbose=2,
                                callbacks=callbacks, validation_split=0.1, shuffle=True)

        # plot loss curve
        loss = results.history['loss']
        val_loss = results.history['val_loss']
        metric = results.history['metric_fun']
        val_metric = results.history['val_metric_fun']
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        x = np.linspace(0, len(loss), len(loss))
        plt.subplot(121), plt.plot(x, loss, x, val_loss)
        plt.title('Loss curve'), plt.legend(['loss', 'val_loss'])
        plt.xlabel('Epochs'), plt.ylabel('loss')
        plt.subplot(122), plt.plot(x, metric, x, val_metric)
        plt.title('metric curve'), plt.legend(['metric', 'val_metric'])
        plt.xlabel('Epochs'), plt.ylabel('ssim')
        plt.show()
        fig.savefig('./evaluation/curve.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def test(self):
        os.makedirs('./evaluation/test_result', exist_ok=True)
        os.makedirs('./evaluation/single picture', exist_ok=True)
        self.unet.load_weights(r'weights/best_model.h5')
        # obtain data
        test_input = scio.loadmat('./mat_data/test_input_retinex.mat')
        test_label = scio.loadmat('./mat_data/test_label_retinex.mat')
        test_num = test_input['input'].shape[0]
        index, step = 0, 0
        n = 0

        while index < test_num:
            print('schedule: %d/%d' % (index, test_num))
            step += 1
            output = self.unet.predict((np.expand_dims(test_input['input'][index:index + 1], axis=3)))
            label = test_label['label'][index]
            result = np.concatenate([test_input['input'][index], output.squeeze(), label], axis=1)
            result = f(result)
            img = Image.fromarray(np.uint8(result * 255))
            img.save('./evaluation/test_result/%d_%.3f_%.3f.png' % (step, ssim(test_input['input'][index], label), ssim(output.squeeze(), label)))
            temp = Image.fromarray(np.uint8(255*f(output.squeeze())))
            temp.save('./evaluation/single picture/%d.png' % step)
            index += 1

    def test_video(self):
        self.unet.load_weights(r'weights/best_model.h5')
        video_input = scio.loadmat('./mat_data/test_video_2.8ml.mat')
        result = self.unet.predict(np.expand_dims(video_input['Video_28'], axis=3))
        # scio.savemat('./evaluation/result_video.mat', {'result_video': result.squeeze()})

if __name__ == '__main__':
    unet = U_Net()
    # unet.train()
    unet.test()
    # unet.test_video()

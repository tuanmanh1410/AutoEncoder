import os
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from skimage import io
import numpy as np

#Show Image
import matplotlib.pyplot as plt

WIDTH = 640 
HEIGH = 480

input_img = Input(shape=(HEIGH, WIDTH, 3))
x = Convolution2D(16, kernel_size = 3, strides= 1, activation='relu', padding='same')(input_img)
x = MaxPooling2D(pool_size = 2, strides = None, padding='same')(x)
x = Convolution2D(8, kernel_size = 3, strides= 1, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size = 2, strides = None, padding='same')(x)
x = Convolution2D(8, kernel_size = 3, strides= 1, activation='relu', padding='same')(x)
encoded = MaxPooling2D(pool_size = 2, strides = None, padding='same')(x)
print("shape of encoded " + str(K.int_shape(encoded)))

#Decode part
x = Convolution2D(8, kernel_size=3, strides=1, activation='relu', padding='same')(encoded)
x = UpSampling2D(size=(2, 2))(x)
x = Convolution2D(8, kernel_size=3, strides=1, activation='relu', padding='same')(x)
x = UpSampling2D(size=(2, 2))(x)
x = Convolution2D(16, kernel_size=3, strides=1, activation='relu', padding='same')(x) 
x = UpSampling2D(size=(2, 2))(x)
decoded = Convolution2D(3, kernel_size=3, strides=1, activation='sigmoid', padding='same')(x)
print ("shape of decoded" + str(K.int_shape(decoded)))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

# Load data
all_images = []
test_images = []

folder = 'data_train/train_Image/'
test_folder = 'data_test/image/'

files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
print("Reading images....")
for i, filename in enumerate(files):
    img = io.imread(filename)
    all_images.append(img)
x_train = np.array(all_images)


test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.jpg')]
print("Reading test images....")
for i, filename in enumerate(test_files):
    img = io.imread(filename)
    test_images.append(img)
x_test = np.array(test_images)


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 480, 640, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 480, 640, 3))  # adapt this if using `channels_first` image data format
print(x_train.shape)
print(x_test.shape)
'''
H = autoencoder.fit_generator(train_generator,
        				steps_per_epoch=int(count/10) + 1,
        				validation_data=test_generator,
        				validation_steps=int(count1/10) + 1,
        				epochs=20, verbose=1)
'''
H = autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=16,
               shuffle=True, validation_data=(x_test, x_test), verbose=1)

autoencoder.save_weights('my_model_weights.h5')

# utility function for showing images
def show_imgs(x_test, decoded_imgs=None, n=2):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(480,640,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decoded_imgs is not None:
            ax = plt.subplot(2, n, i+ 1 +n)
            plt.imshow(decoded_imgs[i].reshape(480,640,3))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

decoded_imgs = autoencoder.predict(x_test)
print("input (upper row)\ndecoded (bottom row)")
show_imgs(x_test, decoded_imgs)







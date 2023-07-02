import tensorflow as tf 
import numpy as np
# import scipy.misc
import imageio
import os

def load_mnist_data(batch_size=64,datasets='mnist',model_name=None):
    assert datasets in ['mnist','fashion_mnist','cifar10','cifar100'], "you should provided a datasets name in 'mnist','fashion_mnist' "
    if datasets=='mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    elif datasets=='fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    # elif datasets=='cifar10':
    #     (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    # elif datasets=='cifar100':
    #     (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    BUFFER_SIZE=train_images.shape[0]
    if model_name=='WGAN' or model_name == 'WGAN_GP':
        train_images = (train_images-127.5)/127.5
    else:
        train_images=(train_images)/255.0
    train_labels=tf.one_hot(train_labels,depth=10)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(batch_size,drop_remainder=True)
    return train_dataset

def inverse_transform(images):
    return (images+1.0)/2.0


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size)) * 255 # imageio.imwrite 不支持将 0-1 之间的数自动 ✖️255 了
    image = image
    image = image.astype(np.uint8)
    return imageio.imwrite(path, image)   # 更新 scipy.misc.imsave 函数为 imageio.imwrite


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

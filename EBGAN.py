from __future__ import division
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, metrics

from ops import *
from utils import *

# the network is based on https://github.com/hwalsuklee/tensorflow-generative-model-collections
class Discriminator(tf.keras.Model):
    def __init__(self, batch_size=64, is_training=True):
        super(Discriminator, self).__init__(name='discriminator')
        self.batch_size = batch_size
        self.is_training = is_training
        self.bn_1 = BatchNorm(is_training=self.is_training)
        self.bn_2 = BatchNorm(is_training=self.is_training)
        self.fc_1 = DenseLayer(32)
        self.fc_2 = DenseLayer(64*14*14)
        self.conv_1 = Conv2D(64, 4, 2)
        self.up_conv_1 = UpConv2D(1, 4, 2)

    def call(self, inputs, training):
        x = self.conv_1(inputs)
        x = layers.ReLU()(x)
        x = layers.Flatten()(x)
        code = self.fc_1(x)
        x = self.fc_2(code)
        x = self.bn_1(x, training)
        x = layers.ReLU()(x)
        x = layers.Reshape((14, 14, 64))(x)
        x = self.up_conv_1(x)
        out = Sigmoid()(x)
        recon_error = tf.math.sqrt(2 * tf.nn.l2_loss(out - inputs)) / self.batch_size
        return out, recon_error, code


class Generator(tf.keras.Model):
    def __init__(self, is_training=True):
        super(Generator, self).__init__(name='generator')
        self.is_training = is_training
        self.fc_1 = DenseLayer(1024)
        self.fc_2 = DenseLayer(128*7*7)
        self.bn_1 = BatchNorm(is_training=self.is_training)
        self.bn_2 = BatchNorm(is_training=self.is_training)
        self.bn_3 = BatchNorm(is_training=self.is_training)
        self.up_conv_1 = UpConv2D(64, 4, 2)
        self.up_conv_2 = UpConv2D(1, 4, 2)

    def call(self, inputs, training):
        x = self.fc_1(inputs)
        x = self.bn_1(x, training)
        x = layers.ReLU()(x)
        x = self.fc_2(x)
        x = self.bn_2(x, training)
        x = layers.ReLU()(x)
        x = layers.Reshape((7, 7, 128))(x)
        x = self.up_conv_1(x)
        x = self.bn_3(x, training)
        x = layers.ReLU()(x)
        x = self.up_conv_2(x)
        x = Sigmoid()(x)
        return x


class EBGAN():
    def __init__(self, args):
        super(EBGAN, self).__init__()
        self.model_name = args.gan_type
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim
        self.sample_z = tf.random.uniform(minval=-1., maxval=1., shape=(
            self.batch_size, self.z_dim), dtype=tf.dtypes.float32)
        self.y_dim = 10
        # margin for loss function
        self.margin = max(1, self.batch_size/64.)
        self.pt_loss_weight = 0.1
        # self.batch_size = 36
        self.checkpoint_dir = check_folder(os.path.join(args.checkpoint_dir, self.model_name))
        self.result_dir = args.result_dir
        self.datasets_name = args.datasets
        self.log_dir = args.log_dir
        self.learnning_rate = args.lr
        self.epoches = args.epoch
        self.datasets = load_mnist_data(datasets=self.datasets_name, batch_size=args.batch_size)
        self.g = Generator(is_training=True)
        self.d = Discriminator(is_training=True)
        self.g_optimizer = keras.optimizers.Adam(lr=5*self.learnning_rate, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(lr=self.learnning_rate, beta_1=0.5)
        self.g_loss_metric = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        self.d_loss_metric = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              generator_optimizer=self.g_optimizer,
                                              discriminator_optimizer=self.d_optimizer,
                                              generator=self.g,
                                              discriminator=self.d)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.datasets_name,
            self.batch_size, self.z_dim)

    def pullaway_loss(self, embeddings):
        """
        Pull Away loss calculation
        :param embeddings: The embeddings to be orthogonalized for varied faces. Shape [batch_size, embeddings_dim]
        :return: pull away term loss
        """
        norm = tf.sqrt(tf.math.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        similarity = tf.matmul(normalized_embeddings,normalized_embeddings, transpose_b=True)
        batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
        pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return pt_loss

    # train for one batch
    @tf.function
    def train_one_step(self, batch_images):
        batch_z = tf.random.uniform(minval=-1,maxval= 1,shape=(self.batch_size, self.z_dim),dtype=tf.dtypes.float32)
        real_images = batch_images
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            D_real_img, D_real_err, D_real_code = self.d(batch_images, training=True)
            fake_imgs = self.g(batch_z, training=True)
            D_fake_img, D_fake_err, D_fake_code = self.d(fake_imgs, training=True)

            # get loss for discriminator
            d_loss = D_real_err + tf.maximum(self.margin - D_fake_err, 0)

            # get loss for generator
            g_loss = D_fake_err + self.pt_loss_weight * self.pullaway_loss(D_fake_code)

        gradients_of_d = d_tape.gradient(d_loss, self.d.trainable_variables)
        gradients_of_g = g_tape.gradient(g_loss, self.g.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients_of_d, self.d.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_g, self.g.trainable_variables))

        self.d_loss_metric(d_loss)
        self.g_loss_metric(g_loss)



    def train(self, load=False):
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log_dir = os.path.join(self.log_dir, self.model_name, current_time)
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.could_load = self.load_ckpt()
        ckpt_step = int(self.checkpoint.step)
        start_epoch = int((ckpt_step*self.batch_size)//60000)

        for epoch in range(start_epoch, self.epoches):
            for batch_images, _ in self.datasets:
                self.train_one_step(batch_images)
                self.checkpoint.step.assign_add(1)
                step = int(self.checkpoint.step)

                # save generated images for every 50 batches training
                if step % 50 == 0:
                    print('step： {}, d_loss: {:.4f}, g_loss: {:.4F}'.format(
                        step, self.d_loss_metric.result(), self.g_loss_metric.result()))
                    manifold_h = int(np.floor(np.sqrt(self.batch_size)))
                    manifold_w = int(np.floor(np.sqrt(self.batch_size)))
                    result_to_display = self.g(self.sample_z, training=False)
                    save_images(result_to_display[:manifold_h * manifold_w, :, :, :],
                                [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(epoch, int(step)))

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('g_loss', self.g_loss_metric.result(), step=step)
                        tf.summary.scalar('d_loss', self.d_loss_metric.result(), step=step)

                #save checkpoints for every 400 batches training
                if step % 400 == 0:
                    save_path = self.manager.save()
                    print("\n----------Saved checkpoint for step {}: {}-----------\n".format(step, save_path))
                    self.g_loss_metric.reset_states()
                    self.d_loss_metric.reset_states()

    def load_ckpt(self):
        self.checkpoint.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("restore model from checkpoint:  {}".format(self.manager.latest_checkpoint))
            return True

        else:
            print("Initializing from scratch.")
            return False


def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gan_type', type=str, default='EBGAN')
    parser.add_argument('--datasets', type=str, default='mnist')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epoch', type=int, default=20,
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62,
                        help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args


def main():
    args = parse_args()
    if args is None:
        exit()
    model = EBGAN(args)
    model.train(load=True)


if __name__ == '__main__':
    main()

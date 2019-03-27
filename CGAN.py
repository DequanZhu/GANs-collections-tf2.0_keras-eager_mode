from __future__ import division
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
from tensorflow import keras
from tensorflow.keras import layers,optimizers,metrics

from ops import *
from utils import *


class CGAN():
    def __init__(self,args):
        super(CGAN, self).__init__()
        self.model_name = args.gan_type
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim
        self.y_dim=10
        self.checkpoint_dir = check_folder(os.path.join(args.checkpoint_dir,self.model_name))
        self.result_dir = args.result_dir
        self.datasets_name = args.datasets
        self.log_dir=args.log_dir
        self.learnning_rate=args.lr
        self.epoches=args.epoch
        self.datasets = load_mnist_data(datasets=self.datasets_name,batch_size=args.batch_size)
        self.g = self.make_generator_model(is_training=True)
        self.d = self.make_discriminator_model(is_training=True)
        self.g_optimizer = optimizers.Adam(lr=5*self.learnning_rate, beta_1=0.5)
        self.d_optimizer = optimizers.Adam(lr=self.learnning_rate, beta_1=0.5)
        self.g_loss_metric = metrics.Mean('g_loss', dtype=tf.float32)
        self.d_loss_metric = metrics.Mean('d_loss', dtype=tf.float32)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              generator_optimizer=self.g_optimizer,
                                              discriminator_optimizer=self.d_optimizer,
                                              generator=self.g,
                                              discriminator=self.d)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)



    # the network is based on https://github.com/hwalsuklee/tensorflow-generative-model-collections
    def make_discriminator_model(self,is_training):
        model = tf.keras.Sequential()
        model.add(Conv2D(64,4,2))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(Conv2D(128,4,2))
        model.add(BatchNorm(is_training=is_training))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Flatten())
        model.add(DenseLayer(1024))
        model.add(BatchNorm(is_training=is_training))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(DenseLayer(1))
        return model


    def make_generator_model(self,is_training):
        model = tf.keras.Sequential()
        model.add(DenseLayer(1024))
        model.add(BatchNorm(is_training=is_training))
        model.add(keras.layers.ReLU())
        model.add(DenseLayer(128*7*7))
        model.add(BatchNorm(is_training=is_training))
        model.add(keras.layers.ReLU())
        model.add(layers.Reshape((7,7,128)))
        model.add(UpConv2D(64,4,2))
        model.add(BatchNorm(is_training=is_training))
        model.add(keras.layers.ReLU())
        model.add(UpConv2D(1,4,2))
        model.add(Sigmoid())
        return model


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.datasets_name,
            self.batch_size, self.z_dim)


    def d_loss_fun(self, d_fake_logits, d_real_logits):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_logits), logits=d_real_logits))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_logits), logits=d_fake_logits))
        total_loss = d_loss_fake+d_loss_real
        return total_loss


    def g_loss_fun(self, logits):
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(logits), logits=logits))
        return g_loss


    # train for one batch
    @tf.function
    def train_one_step(self,batch_images,batch_labels):
        z = np.random.uniform(-1, 1,[self.batch_size, self.z_dim]).astype(np.float32)
        z_y = tf.concat([z, batch_labels], 1)
        real_images = conv_cond_concat(batch_images, batch_labels)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_imgs = self.g(z_y, training=True)
            fake_imgs = conv_cond_concat(fake_imgs, batch_labels)
            d_fake_logits = self.d(fake_imgs, training=True)
            d_real_logits = self.d(real_images, training=True)
            d_loss = self.d_loss_fun(d_fake_logits, d_real_logits)
            g_loss = self.g_loss_fun(d_fake_logits)
        gradients_of_d = d_tape.gradient(d_loss, self.d.trainable_variables)
        gradients_of_g = g_tape.gradient(g_loss, self.g.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients_of_d, self.d.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_g, self.g.trainable_variables))
        self.g_loss_metric(g_loss)
        self.d_loss_metric(d_loss)



    def train(self, load=False):
        self.sample_label=np.random.randint(0,self.y_dim-1,size=(self.batch_size))
        self.sample_label=tf.one_hot(self.sample_label,depth=10)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(self.log_dir, self.model_name, current_time)
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # if want to load a checkpoints,set load flags to be true
        if load:
            self.could_load = self.load_ckpt()
            ckpt_step = int(self.checkpoint.step)
            start_epoch=int((ckpt_step*self.batch_size)//60000)
        else:
            start_epoch=0


        for epoch in range(start_epoch,self.epoches):
            for batch_images, batch_labels in self.datasets:
                self.train_one_step(batch_images,batch_labels)
                self.checkpoint.step.assign_add(1)
                step = int(self.checkpoint.step)

                # save generated images for every 50 batches training
                if step % 50 == 0:
                    print ('step： {}, d_loss: {:.4f}, g_oss: {:.4F}'.format(step,self.d_loss_metric.result(), self.g_loss_metric.result()))
                    manifold_h = int(np.floor(np.sqrt(self.batch_size)))
                    manifold_w = int(np.floor(np.sqrt(self.batch_size)))
                    sample_z = np.random.uniform(-1., 1., size=(self.batch_size, self.z_dim)).astype(np.float32)
                    sample_z_y=tf.concat([sample_z,self.sample_label],1)
                    result_to_display = self.g(sample_z_y, training=False)
                    save_images(result_to_display[:manifold_h * manifold_w, :, :, :],
                                [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(epoch, int(step)))

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('g_loss', self.g_loss_metric.result(), step=step)
                        tf.summary.scalar('d_loss', self.d_loss_metric.result(), step=step)


                #save checkpoints for every 400 batches training
                if step % 400 == 0:
                    save_path = self.manager.save()
                    
                    print("\n----------Saved checkpoint for step {}: {}-------------\n".format(step, save_path))

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
    parser.add_argument('--gan_type', type=str, default='CGAN')
    parser.add_argument('--datasets', type=str, default='mnist')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
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
    model = CGAN(args)
    model.train(load=True)


if __name__ == '__main__':
    main()
 

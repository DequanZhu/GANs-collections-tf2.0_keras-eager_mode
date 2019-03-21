from __future__ import division
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
from tensorflow import keras
from ops2 import *
from utils1 import *
# from tensorflow.keras import layers


class Generator(tf.keras.Model):
    def __init__(self, z_dim, y_dim, is_training=True):
        super(Generator, self).__init__(name='generator')
        self.input_dim = z_dim+y_dim
        self.fc_1 = DenseLayer(1024, input_dim=self.input_dim,)
        self.fc_2 = DenseLayer(128*7*7)
        self.reshape = keras.layers.Reshape((7, 7, 128))
        self.up_1 = Upsample(64, (4, 4), apply_batchnorm=True)
        self.up_2 = Upsample(1, (4, 4), apply_batchnorm=False)

    def call(self, inputs, training):
        x = self.fc_1(inputs, training=training)
        x = self.fc_2(x, training=training)
        x = self.reshape(x)
        x = self.up_1(x, training=training)
        x = self.up_2(x, training=training)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, is_training=True):
        super(Discriminator, self).__init__(name='discriminator')
        self.down_1 = Downsample(64, (4, 4), apply_batchnorm=False)
        self.down_2 = Downsample(128, (4, 4), apply_batchnorm=True)
        self.fc_1 = DenseLayer(1024, activation='lrelu')
        self.flatten_1 = keras.layers.Flatten()
        self.fc_2 = DenseLayer(1, apply_batchnorm=False)

    def call(self, inputs, training):
        x = self.down_1(inputs, training=training)
        x = self.down_2(x, training=training)
        x = self.flatten_1(x)
        x = self.fc_1(x, training=training)
        out_logits = self.fc_2(x, training=training)
        out = keras.activations.sigmoid(out_logits)
        return out, out_logits, x


class CGAN():
    def __init__(self):
        super(CGAN, self).__init__()
        self.model_name = 'CGAN'
        self.batch_size = 64
        self.z_dim = 62
        self.y_dim = 10
        self.num_examples_to_generate = 36
        self.checkpoint_dir = './tf_ckpts'
        self.result_dir = './results'
        self.dataset_name = 'mnist'
        self.log_dir='./logs/'
        self.datasets = load_mnist_data()
        self.g = Generator(self.z_dim, self.y_dim, is_training=True)
        self.d = Discriminator(is_training=True)
        self.g_optimizer = keras.optimizers.Adam(lr=1e-3, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(lr=2e-4, beta_1=0.5)
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
            self.model_name, self.dataset_name,
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

    def train_one_step(self,batch_labels, batch_images):
        noises = np.random.uniform(-1, 1,[self.batch_size, self.z_dim]).astype(np.float32)
        batch_z = tf.concat([noises, batch_labels], 1)
        real_images = conv_cond_concat(batch_images, batch_labels)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_imgs = self.g(batch_z, training=True)
            fake_imgs = conv_cond_concat(fake_imgs, batch_labels)
            d_fake, d_fake_logits, _ = self.d(fake_imgs, training=True)
            d_real, d_real_logits, _ = self.d(real_images, training=True)
            d_loss = self.d_loss_fun(d_fake_logits, d_real_logits)
            g_loss = self.g_loss_fun(d_fake_logits)
        gradients_of_d = d_tape.gradient(d_loss, self.d.trainable_variables)
        gradients_of_g = g_tape.gradient(g_loss, self.g.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients_of_d, self.d.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_g, self.g.trainable_variables))
        self.g_loss_metric(g_loss)
        self.d_loss_metric(d_loss)



    def train(self, epoches, load=False):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = self.log_dir+self.model_name+'/'+ current_time
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.could_load = self.load_ckpt()
        ckpt_step = int(self.checkpoint.step)
        start_epoch=int((ckpt_step*self.batch_size)//60000)


        for epoch in range(start_epoch,epoches):
            for batch_images, batch_labels in self.datasets:
                if int(self.checkpoint.step) == 0 or self.could_load:
                    self.could_load = False
                    sample_z = np.random.uniform(-1, 1, size=(self.num_examples_to_generate, self.z_dim))
                    test_labels = batch_labels[0:self.num_examples_to_generate]
                    self.batch_z_to_disply = tf.concat([sample_z, test_labels[:self.num_examples_to_generate, :]], 1)
                self.train_one_step(batch_labels, batch_images)
                self.checkpoint.step.assign_add(1)
                step = int(self.checkpoint.step)


                if step % 50 == 0:
                    print ('step： {}, d_loss: {:.4f}, g_oss: {:.4F}'.format(step,self.d_loss_metric.result(), self.g_loss_metric.result()))
                    manifold_h = int(np.floor(np.sqrt(self.num_examples_to_generate)))
                    manifold_w = int(np.floor(np.sqrt(self.num_examples_to_generate)))
                    result_to_display = self.g(self.batch_z_to_disply, training=False)
                    save_images(result_to_display[:manifold_h * manifold_w, :, :, :],
                                [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(epoch, int(step)))

                    with self.train_summary_writer.as_default():
                        print("-----------write to summary-----------")
                        tf.summary.scalar('g_loss', self.g_loss_metric.result(), step=step)
                        tf.summary.scalar('d_loss', self.d_loss_metric.result(), step=step)

                if step % 200 ==0:
                    save_path = self.manager.save()
                    
                    print("\n---------------Saved checkpoint for step {}: {}------------------\n".format(step, save_path))

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



if __name__ == '__main__':
    model = CGAN()
    model.train(epoches=20, load=True)

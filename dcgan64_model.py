# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_integer('image_size', 128,
                     'Image size. Default is 128')
flags.DEFINE_integer('df_dim', 64,
                     'Number of filters for discriminator')
flags.DEFINE_integer('gf_dim', 64,
                     'Number of filters for generator')
flags.DEFINE_integer('c_dim', 3,
                     'Number of image channels')

image_size = FLAGS.image_size
s16 = image_size // 32
df_dim = FLAGS.df_dim
gf_dim = FLAGS.gf_dim
c_dim = FLAGS.gf_dim


def _leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)


def _batch_norm(x, is_training, name):
    return tf.layers.batch_normalization(
        x, momentum=0.9, epsilon=1e-5, training=is_training, name=name)


def _dense(x, channels, name):
    return tf.layers.dense(
        x, channels,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        name=name)


def _conv2d(x, filters, kernel_size, stride, name):
    return tf.layers.conv2d(
        x, filters, [kernel_size, kernel_size],
        strides=[stride, stride], padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        name=name)


def _deconv2d(x, filters, kernel_size, stride, name):
    return tf.layers.conv2d_transpose(
        x, filters, [kernel_size, kernel_size],
        strides=[stride, stride], padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        name=name)


def discriminator(x, is_training=True, scope='Discriminator'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = _conv2d(x, df_dim, 5, 2, name='d_conv1')
        x = _leaky_relu(x)

        x = _conv2d(x, df_dim*2, 5, 2, name='d_conv2')
        x = _leaky_relu(_batch_norm(x, is_training, name='d_bn2'))

        x = _conv2d(x, df_dim*4, 5, 2, name='d_conv3')
        x = _leaky_relu(_batch_norm(x, is_training, name='d_bn3'))

        x = _conv2d(x, df_dim*8, 5, 2, name='d_conv4')
        x = _leaky_relu(_batch_norm(x, is_training, name='d_bn4'))

        x = tf.reshape(x, [-1, s16, s16, df_dim*8])

        x = _dense(x, 1, name='d_fc_4')

        return x


def generator(x, is_training=True, scope='Generator'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = _dense(x, gf_dim * 8 * s16 * s16, name='g_fc1')
        x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn1'))

        x = tf.reshape(x, [-1, s16, s16, gf_dim*8], name="reshape_1")

        x = _deconv2d(x, gf_dim * 4, 5, 2, name='g_dconv2')
        x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn2'))

        x = _deconv2d(x, gf_dim * 2, 5, 2, name='g_dconv3')
        x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn3'))

        x = _deconv2d(x, gf_dim, 5, 2, name='g_dconv4')
        x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn4'))

        x = _deconv2d(x, c_dim, 5, 2, name='g_dconv5')
        x = tf.tanh(x)

        return x

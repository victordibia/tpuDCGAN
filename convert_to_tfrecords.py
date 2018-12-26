"""
Some codes from https://github.com/Newmu/dcgan_code
Some code from 
Convert images in a folder to TFRecords
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import io
from random import shuffle

import utils.imutils as imutils
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def convert_folder_tf_sub(data_dir, output_file, display_images, image_size):
#     index = 0
#     dir_list = []
#     dir_count = 0
#     all_file_list = []
#     with tf.python_io.TFRecordWriter(output_file) as record_writer:
#         for root, dirs, filenames in os.walk(data_dir):
#             for dir in dirs:
#                 for f in os.listdir(os.path.join(data_dir, dir)):
#                     file_path = os.path.join(data_dir, dir, f)
#                     if file_path.split(".")[1] == "jpg" or file_path.split(".")[1] == "png":
#                         all_file_list.append(file_path)
#         print("BEfore shuffle", len(all_file_list))
#         shuffle(all_file_list)
#         new_list = all_file_list[:3300]
#         # print((new_list))
#         for file_path in new_list:
#             index += 1
#             image = imutils.get_image(
#                 file_path, image_size, is_crop=True, resize_w=image_size)
#             image = imutils.colorize(image)
#             assert image.shape == (image_size, image_size, 3)
#             if (image.shape == (image_size, image_size, 3)):
#                 image += 1.
#                 image *= (255. / 2.)
#                 image = image.astype('uint8')
#                 image_raw = image.tostring()
#                 if (display_images == 1):
#                     plt.imshow(image)
#                     plt.show()
#                 bytes_feat = _bytes_feature(image_raw)
#                 print(index, " Writing to tfrecord | ",
#                       " Image size: ", image_size, " Directory: ", file_path)
#                 example = tf.train.Example(features=tf.train.Features(feature={
#                     'image': _bytes_feature(image_raw),
#                     'label': _int64_feature(dir_count)
#                 }))
#                 serialized_example = example.SerializeToString()
#                 record_writer.write(serialized_example)
#         print("Done!")


def convert_folder_tf(data_dir, output_file, display_images, image_size):
    index = 0
    dir_list = []
    dir_count = 0
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for root, dirs, filenames in os.walk(data_dir):
            for dir in dirs:
                if dir not in dir_list:
                    dir_list.append(dir)
                    dir_count += 1
                for f in os.listdir(os.path.join(data_dir, dir)):
                    file_path = os.path.join(data_dir, dir, f)
                    if file_path.split(".")[1] == "jpg" or file_path.split(".")[1] == "png":
                        index += 1
                        image = imutils.get_image(
                            file_path, image_size, is_crop=True, resize_w=image_size)
                        image = imutils.colorize(image)
                        assert image.shape == (image_size, image_size, 3)
                        if (image.shape == (image_size, image_size, 3)):
                            image += 1.
                            image *= (255. / 2.)
                            image = image.astype('uint8')
                            image_raw = image.tostring()
                            if (display_images == 1):
                                plt.imshow(image)
                                plt.show()
                            bytes_feat = _bytes_feature(image_raw)
                            print(index, " Writing to tfrecord | ",
                                  " Image size: ", image_size, " Directory: ", dir)
                            example = tf.train.Example(features=tf.train.Features(feature={
                                'image': _bytes_feature(image_raw),
                                'label': _int64_feature(dir_count)
                            }))
                            serialized_example = example.SerializeToString()
                            record_writer.write(serialized_example)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='images/cifar',
        help='Directory containing images')
    parser.add_argument(
        '--output_file',
        type=str,
        default='images/train.tfrecords',
        help='Directory to save tfrecords')
    parser.add_argument(
        '--image_size',
        type=int,
        default=64,
        help='Resize image')
    parser.add_argument(
        '--display_images',
        type=int,
        default=0,
        help='To display during conversion')

    args = parser.parse_args()
    print(args)
    convert_folder_tf(args.data_dir, args.output_file,
                      args.display_images, args.image_size)

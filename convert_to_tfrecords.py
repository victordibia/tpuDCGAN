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
import scipy.misc
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


 

def imread(path):
    img = scipy.misc.imread(path)
    if len(img.shape) == 0:
        raise ValueError(path + " got loaded as a dimensionless array!")
    return img.astype(np.float)

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    h, w = x.shape[:2]
    crop_h = min(h, w)  # we changed this to override the original DCGAN-TensorFlow behavior
                        # Just use as much of the image as possible while keeping it square
    if crop_w is None:
        crop_w = crop_h
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    cropped_image = center_crop(image, npx, resize_w=resize_w)
    return np.array(cropped_image)/127.5 - 1.

def get_image(image_path, image_size, is_crop=True, resize_w=64):
    global index
    out = transform(imread(image_path), image_size, is_crop, resize_w)
    return out

def colorize(img):
    if img.ndim == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = np.concatenate([img, img, img], axis=2)
    if img.shape[2] == 4:
        img = img[:, :, 0:3]
    return img

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
                        image = get_image(
                            file_path, image_size, is_crop=True, resize_w=image_size)
                        image = colorize(image)
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
                            print(index, " Writing to tfrecord | ", " Image size: ", image_size, " Directory: ", dir)
                            example = tf.train.Example(features=tf.train.Features(feature={
                                'image': _bytes_feature(image_raw),
                                'label': _int64_feature(dir_count)
                            }))
                            serialized_example = example.SerializeToString()
                            record_writer.write(serialized_example)
 


 

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
    convert_folder_tf(args.data_dir, args.output_file, args.display_images, args.image_size)

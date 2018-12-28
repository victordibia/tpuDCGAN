"""
Functions to help extract features from an image using VGG.
Compute cosine distances etc
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import imutils
import matplotlib.pyplot as plt
from scipy import spatial

tf.enable_eager_execution()

layer_list = ["block1_pool", "block2_pool",
              "block3_pool", "block4_pool", "block5_pool"]

model = VGG16(weights='imagenet', include_top=False)


def get_intermediate_models(layer_list):

    intermediate_models = []
    for layer in layer_list:
        intermediate_model = get_intermediate_model(layer)
        intermediate_models.append(
            {"model": intermediate_model, "name": layer})
        # intermediate_model.summary()
    tf.logging.info("Finished generating", len(
        intermediate_models), " intermediate models")
    return intermediate_models


def get_intermediate_model(layer_name):
    return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# Get NP array of image


def get_np_array_from_image(img_path):
    image_size = 64
    loaded_image = imutils.get_image(
        img_path, image_size, is_crop=True, resize_w=image_size)
    loaded_image = imutils.colorize(loaded_image)
    return loaded_image


# image1 = get_np_array_from_image("images/masks/novicmasks/9.jpg")
# image2 = get_np_array_from_image("images/masks/novicmasks/51.jpg")
# # # plt.imshow(imutils.convert_array_to_image(image1))
# # # plt.show()
# # print(image1.shape)


def get_features(model, image_data):
    return np.array(model.predict(np.expand_dims(image_data, axis=0))).flatten()


# Compute cosine distance between two vectors
def compute_cosine_distance(feat_1, feat_2):
    cosine_similarity = 1 - spatial.distance.cosine(feat_1, feat_2)
    return cosine_similarity

# Compute cosine distance between a vector and elements in a matrix


def compute_cosine_distance_matrix(feat, feat_matrix):
    cosine_dist_matrix = spatial.distance.cdist(
        feat_matrix, feat.reshape(1, -1), 'cosine').reshape(-1, 1)
    return 1 - cosine_dist_matrix


image_size = 64

# Parse tfrecord file


def tfrecord_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image_data = tf.decode_raw(features['image'], tf.uint8)
    image_data.set_shape([image_size*image_size*3])
    # Normalize the values of the image from the range [0, 255] to [-1.0, 1.0]
    image_data = tf.cast(image_data, tf.float32) * (2.0 / 255) - 1.0
    # image = tf.transpose(tf.reshape(image, [3, 32*32]))
    label = tf.cast(features['label'], tf.int32)
    # print(label)
    return image_data, label

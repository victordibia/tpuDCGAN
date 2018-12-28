import matplotlib.pyplot as plt
from PIL import Image
import scipy.spatial
import feature_extractor as feat_ex
import numpy as np
import math
import tensorflow as tf
import json
import imutils


layer_name = "block5_pool"
intermediate_model = feat_ex.get_intermediate_model(layer_name)

original_images = []
original_images_feats = []

image_size = 64
trunc_size = 0
index_search = 55
num_similar = 24  # top n most similar images.
tfrecord_file_path = "images/masks/train_masks_64.tfrecords"
np_zarray_location = "test/features_" + layer_name + "0.npz"


def load_images_from_tfrecord():

    tf.logging.info("Features looaded from file", len(original_images_feats))
    read_records = tf.python_io.tf_record_iterator(tfrecord_file_path)
    for i, s_example in enumerate(read_records):
        image_data, label = feat_ex.tfrecord_parser(s_example)
        image_data = tf.reshape(image_data, [image_size, image_size, 3])
        original_images.append(np.array(image_data))

        # Read only trunc_size amount of values from tf_record
        if (trunc_size != 0 and i == trunc_size):
            break

    tf.logging.info("> Loaded images from tfrecord", len(original_images))


# load features from Numpy compressed file


def load_features_from_npz():
    global original_images_feats
    # load features from Numpy compressed file
    feats = np.load(np_zarray_location)
    original_images_feats = feats["feat_array"]
    tf.logging.info("> Loaded features from npz with shape",
                    original_images_feats.shape)


def plot_similar(image, similarity_score_index, similar_images, similarity_scores):

    f_image = imutils.convert_array_to_image(image)
    fig = plt.figure(figsize=(10, 10))
    columns = rows = math.sqrt(len(similarity_score_index)) + 1
    # rows = column
    ax = fig.add_subplot(rows, columns, 1)
    ax.set_title("Main Image", fontweight="bold",
                 size=10)
    plt.imshow(f_image)

    for i in range(0, len(similarity_score_index)):
        ax = fig.add_subplot(rows, columns, i+2)
        ax.set_title(
            "img" + str(similarity_score_index[i]) + " : " +
            str(round(similarity_scores[i], 3)),
            size=10)
        curr_img = imutils.convert_array_to_image(similar_images[i])
        plt.imshow(curr_img)
    fig.tight_layout()
    plt.show()


load_features_from_npz()
load_images_from_tfrecord()

# matrix containing similarity scores for all images in tfrecord
feat_matrix = original_images_feats
# print(feat_matrix.shape)
np_original = np.array(original_images)
unique_images = []

# Extract similarity scores for
# file with source_id


def get_similarity_list(source_id):
    global unique_images

    source_image_data = feat_ex.get_np_array_from_image(
        "test/generated/" + str(source_id) + ".jpg")
    # get features for source image
    feat = feat_ex.get_features(intermediate_model, source_image_data)

    # get similarity measure with entire dataset
    cosine_dist_matrix = feat_ex.compute_cosine_distance_matrix(
        feat, feat_matrix)
    similarity_score_index = cosine_dist_matrix.flatten().argsort()[
        ::-1][:num_similar]
    similarity_scores = cosine_dist_matrix[similarity_score_index].flatten()

    similar_images = np_original[similarity_score_index]

    # plot_similar(source_image_data, similarity_score_index,
    #              similar_images, similarity_scores)
    similarity_details = []
    for i in range(len(similar_images)):
        # keep track of images we havent seen
        if (similarity_score_index[i] not in unique_images):
            unique_images.append(similarity_score_index[i])

        similarity_dict = {"id": int(similarity_score_index[i]),
                           "score": (round(similarity_scores[i], 4)),
                           "layer": layer_name,
                           "sourceid": source_id
                           }
        similarity_details.append(similarity_dict)

    # print(similarity_details)
    print(source_id, " > processing similarity details using VGG layer ", layer_name)
    return similarity_details


all_similarity_details = {}
for i in range(100):
    all_similarity_details[i] = get_similarity_list(i)

# Save similarity data to json file to be used in demo
with open("test/demo/" + layer_name + ".json", 'w') as fp:
    json.dump(all_similarity_details, fp)

# Save all similar images to a location
for im in unique_images:
    img = imutils.convert_array_to_image(original_images[im])
    img.save("test/demo/images/" + str(im) + ".jpg")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import imutils
import matplotlib.pyplot as plt
from scipy import spatial

model = VGG16(weights='imagenet', include_top=False)


def get_intermediate_models(model, layer_list):
    intermediate_models = []
    for layer in layer_list:
        intermediate_model = Model(inputs=model.input,
                                   outputs=model.get_layer(layer).output)
        intermediate_models.append(
            {"model": intermediate_model, "name": layer})
        # intermediate_model.summary()
    tf.logging.info("Finished generating", len(
        intermediate_models), " intermediate models")
    return intermediate_models


layer_list = ["block1_pool", "block2_pool",
              "block3_pool", "block4_pool", "block5_pool"]
intermediate_models = get_intermediate_models(model, layer_list)


def get_np_array_from_image(img_path):
    image_size = 64
    loaded_image = imutils.get_image(
        img_path, image_size, is_crop=True, resize_w=image_size)
    loaded_image = imutils.colorize(loaded_image)
    # print(loaded_image)
    return loaded_image


image1 = get_np_array_from_image("images/masks/novicmasks/9.jpg")
image2 = get_np_array_from_image("images/masks/novicmasks/51.jpg")
# plt.imshow(imutils.convert_array_to_image(image1))
# plt.show()
print(image1.shape)


def compute_distance(image1, image2, model):
    feat_1 = np.array(model.predict(np.expand_dims(image1, axis=0))).flatten()
    feat_2 = np.array(model.predict(np.expand_dims(image2, axis=0))).flatten()
    cosine_similarity = 1 - spatial.distance.cosine(feat_1, feat_2)
    return cosine_similarity


for i, model in enumerate(intermediate_models):
    print(i, model["name"], compute_distance(image1, image2, model["model"]))


# vgg16_feature = model.predict(img_data)

# print(vgg16_feature.shape)

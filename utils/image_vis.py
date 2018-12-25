import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np


model = VGG16(weights='imagenet', include_top=False)
# model.summary()

# mp1_layer = 'block2_pool'
# mp1_model = Model(inputs=model.input,
#                   outputs=model.get_layer(mp1_layer).output)

# mp1_model.summary()


def get_intermediate_models(model, layer_list):
    intermediate_models = []
    for layer in layer_list:
        intermediate_model = Model(inputs=model.input,
                                   outputs=model.get_layer(layer).output)
        intermediate_models.append(intermediate_model)
        # intermediate_model.summary()
    return intermediate_models


layer_list = ["block1_pool", "block2_pool",
              "block3_pool", "block4_pool", "block5_pool"]
# get_intermediate_models(model, layer_list)

img_path = 'images/masks/novicmasks/1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
print(img_data.shape)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

print(img_data.shape)

# vgg16_feature = model.predict(img_data)

# print(vgg16_feature.shape)

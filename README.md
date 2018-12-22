## Train a GAN using TPUs and Tensorflow on Google Cloud

This repo contains code to train an unconditional DCGAN using TPUs on Google Cloud. It is based on the experimental TPU examples with the following modifications  

- Support for `64*64` and `128*128` generation: Provide two model architectures (mainly additional layers) that support generating higher resolution images (64, 128).
- Images to TFRecords: A script is available to convert images in a folder to TFRecords required to train the DCGAN.
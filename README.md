## Train a GAN using TPUs and Tensorflow on Google Cloud

This repo contains code to train an unconditional DCGAN using TPUs on Google Cloud. It is based on the experimental TPU examples with the following modifications  

- Support for `64*64` and `128*128` generation: Provide two model architectures (mainly additional layers) that support generating higher resolution images (64, 128).
- Images to TFRecords: A script is available to convert images in a folder to TFRecords required to train the DCGAN.

## Convert Images

The `convert_to_tfrecords` script accepts arguments for data directory (`data_dir`) and output file (`output_file`). Data directory is expected to have folders which contain images directly.

```shell
python convert_to_tfrecords
```

## Training

- Please follow the official tensorflow tutorial on setting up a TPU instance. 
- Clone this repo
```shell
git clone https://github.com/victordibia/tpuDCGAN
```
- Start Training
```shell
export GCS_BUCKET_NAME=  <Your GCS Bucket>
python dcgan_main.py --tpu=$TPU_NAME --train_data_file=gs://$GCS_BUCKET_NAME/data/masks/train_masks.tfrecords   --dataset=dcgan64 --train_steps=10000 --train_steps_per_eval=500 --model_dir=gs://$GCS_BUCKET_NAME/dcgan/masks/model --test_data_file=gs://$GCS_BUCKET_NAME/data/rand/test.tfrecords

```
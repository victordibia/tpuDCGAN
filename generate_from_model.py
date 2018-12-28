"""
Generate images using a model checkpoint
"""

import utils.imutils as imutils
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.estimator import estimator
import argparse

import dcgan64_input
import dcgan64_model
import dcgan128_input
import dcgan128_model


noise_dim = 100
current_step = 0
model = None


def noise_input_fn(params):
    """Input function for generating samples for PREDICT mode.

    Generates a single Tensor of fixed random noise. Use tf.data.Dataset to
    signal to the estimator when to terminate the generator returned by
    predict().

    Args:
      params: param `dict` passed by TPUEstimator.

    Returns:
      1-element `dict` containing the randomly generated noise.
    """
    np.random.seed(50)
    noise_dataset = tf.data.Dataset.from_tensors(tf.constant(
        np.random.randn(params['batch_size'], noise_dim), dtype=tf.float32))
    noise = noise_dataset.make_one_shot_iterator().get_next()
    return {'random_noise': noise}, None


def model_fn(features, labels, mode, params):
    """Constructs DCGAN from individual generator and discriminator networks."""
    del labels    # Unconditional GAN does not use labels

    if mode == tf.estimator.ModeKeys.PREDICT:
        ###########
        # PREDICT #
        ###########
        # Pass only noise to PREDICT mode
        random_noise = features['random_noise']
        predictions = {
            'generated_images': model.generator(random_noise, is_training=False)
        }

        return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)


def generate_images(image_size, model_dir, output_dir):
    global model
    if(image_size == 64):
        model = dcgan64_model
    else:
        model = dcgan128_model

    config = tf.contrib.tpu.RunConfig(
        # cluster=tpu_cluster_resolver,
        model_dir=model_dir)

    _NUM_VIZ_IMAGES = 100
    cpu_est = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=False,
        config=config,
        predict_batch_size=_NUM_VIZ_IMAGES)

    # Render  generated images
    generated_iter = cpu_est.predict(
        input_fn=noise_input_fn, checkpoint_path=model_dir)
    images = [p['generated_images'][:, :, :] for p in generated_iter]
    assert len(images) == _NUM_VIZ_IMAGES

    image_rows = [np.concatenate(images[i:i+10], axis=0)
                  for i in range(0, _NUM_VIZ_IMAGES, 10)]
    tiled_image = np.concatenate(image_rows, axis=1)

    img = imutils.convert_array_to_image(tiled_image)

    step_string = str(current_step).zfill(5)
    file_obj = tf.gfile.Open(
        os.path.join(output_dir, 'gen.png'), 'w')
    img.save(file_obj, format='png')
    tf.logging.info('Finished generating images')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='models/masks/64/model.ckpt-14400',
        help='Directory containing model checkpoint')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/masks/64/',
        help='Directory to save generated images')
    parser.add_argument(
        '--image_size',
        type=int,
        default=64,
        help='Model to use for generation. Valid values are  [64,128]')
    # parser.add_argument(
    #     '--display_images',
    #     type=int,
    #     default=0,
    #     help='To display during conversion')

    args = parser.parse_args()
    print(args)
    generate_images(args.image_size, args.model_dir, args.output_dir)

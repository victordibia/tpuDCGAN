import scipy.misc
import utils.imutils as imutils


def imread(path):
    img = scipy.misc.imread(path)
    if len(img.shape) == 0:
        raise ValueError(path + " got loaded as a dimensionless array!")
    return img.astype(np.float)


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    h, w = x.shape[:2]
    # we changed this to override the original DCGAN-TensorFlow behavior
    crop_h = min(h, w)
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

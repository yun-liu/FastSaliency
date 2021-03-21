import numpy as np
import torch
import random
import cv2
import numbers
import functional as F
import math
from PIL import Image


class Scale(object):
    def __init__(self, wi, he):
        self.w = wi
        self.h = he

    def __call__(self, image, label):
        # bilinear interpolation for RGB image
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        # nearest neighbour interpolation for label image
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return [image, label]


class RandomCropResize(object):
    def __init__(self, crop_area):
        self.c = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.c)
            y1 = random.randint(0, self.c)

            img_crop = img[y1:h - y1, x1:w - x1]
            label_crop = label[y1:h - y1, x1:w - x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
            return img_crop, label_crop
        else:
            return [img, label]


class RandomFlip(object):
    def __call__(self, image, label):
        if random.random() < 0.5:
            x1 = 0 # random.randint(0, 1) # if you want to do vertical flip, uncomment this line
            if x1 == 0:
                image = cv2.flip(image, 0) # horizontal flip
                label = cv2.flip(label, 0) # horizontal flip
            else:
                image = cv2.flip(image, 1) # veritcal flip
                label = cv2.flip(label, 1) # veritcal flip
        return [image, label]


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        image = (image - self.mean) / self.std
        label /= 255.

        return [image, label]


class RandomCrop(object):
    def __init__(self, scale_factor=(0.8, 1.)):
        self.scale_factor = scale_factor

    @staticmethod
    def get_params(image, scale_factor):
        scale = random.uniform(*scale_factor)
        h, w = image.shape[:2]
        size = int(min(h, w) * scale)

        i = random.randint(0, h - size)
        j = random.randint(0, w - size)
        return i, j, size

    def __call__(self, image, label):
        orig_h, orig_w = image.shape[:2]
        i, j, size = self.get_params(image, self.scale_factor)

        image = cv2.resize(image[i:i + size, j:j + size], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label[i:i + size, j:j + size], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return [image, label]


class RandomRotate(object):
    def __init__(self, degree=0):
        self.degree = (-degree, degree)

    @staticmethod
    def get_params(image, degree):
        deg = random.uniform(*degree)
        if deg < 0:
            deg = 360. + deg

        h, w = image.shape[:2]
        new_w, new_h = largest_rotated_rect(w, h, math.radians(deg))
        return deg, new_w, new_h

    def __call__(self, image, label):
        orig_h, orig_w = image.shape[:2]
        deg, w, h = self.get_params(image, self.degree)

        image_rotated = rotate_image(image, deg, flags=cv2.INTER_LINEAR)
        label_rotated = rotate_image(label, deg, flags=cv2.INTER_NEAREST)

        image_rotated = crop_around_center(image_rotated, w, h)
        label_rotated = crop_around_center(label_rotated, w, h)

        image_rotated = cv2.resize(image_rotated, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        label_rotated = cv2.resize(label_rotated, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return [image_rotated, label_rotated]


class ToTensor(object):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (int(w / self.scale), int(h / self.scale)), interpolation=cv2.INTER_NEAREST)
        image = image.transpose((2, 0, 1))

        image_tensor = torch.from_numpy(image)
        label_tensor = torch.ByteTensor(label.astype(np.uint8))

        return [image_tensor, label_tensor]


class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + ' object is not callable'
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class ColorJitter(object):
    '''Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    '''
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError('If {} is a single number, it must be non negative.'.format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError('{} values should be between {}'.format(name, bound))
        else:
            raise TypeError('{} should be a single number or a list/tuple with lenght 2.'.format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        '''Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        '''
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, image, label):
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return [transform(image), label]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            if isinstance(args, (tuple, list)):
                args = t(*args)
            else:
                args = t(args)
        return args


def rotate_image(image, angle, flags):
    '''
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    '''
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
            )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
            (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
            ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
            ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=flags)

    return result


def largest_rotated_rect(w, h, angle):
    '''
    Given a rectangle of size (w, h) that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    '''
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)


def crop_around_center(image, width, height):
    '''
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    '''
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

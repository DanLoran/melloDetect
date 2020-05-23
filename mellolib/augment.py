from math import log
from PIL import Image, ImageOps
import numpy as np

def rotator(num_images_total):
    def f(image, i):
        rotation_amount = 360 / num_images_total
        image.rotate(rotation_amount * i)
    f.num = num_images_total
    return f

# https://forums.blinkstick.com/t/python-function-for-color-temperature/1068
def color_temp(temp):
    temp = temp / 100
    if temp <= 66:
        r = 255
    else:
        r = temp - 60
        r = 329.698727446 * (r ** -0.1332047592)
        r = min(255, max(0, r))

    if temp < 66:
        g = temp
        g = 99.4708025861 * log(g) - 161.1195681661
        g = min(255, max(0, g))
    else:
        g = temp - 60
        g = 288.1221695283 * (g ** -0.0755148492)
        g = min(255, max(0, g))

    if temp >= 65:
        b = 255
    elif temp < 20:
        b = 0
    else:
        b = temp - 10
        b = 138.5177312231 * log(b) - 305.0447927307
        b = min(255, max(0, b))
    return (r/255, g/255, b/255)

# candlelight is 1900k
# direct sunlight (no color balancing) is 6000k
# blue sky is 20000k
# recommended is [1900, 15000]
def white_balancer(temps):
    def f(image, i):
        image_data = np.array(image.convert('RGB')).astype(np.uint16)
        channel_offsets = color_temp(temps[i])
        transformed_image_data = np.moveaxis(np.array(
                [image_data[:, :, 0] * channel_offsets[0],
                 image_data[:, :, 1] * channel_offsets[1],
                 image_data[:, :, 2] * channel_offsets[2]]).clip(0, 255).astype(np.uint8), 0, 2)
        Image.fromarray(transformed_image_data)
    f.num = len(temps)
    return f

def mirrorer():
    def f(image, i):
        if i == 0:
            return ImageOps.flip(image)
        elif i == 1:
            return ImageOps.mirror(image)
        else:
            raise Exception("Invalid index in mirrorer: " + i)
    f.num = 2
    return f

def noiser(num_images_total, noise_magnitude):
    def f(image, i):
        np.random.seed(i)
        image_data = np.array(image.convert('RGB')).astype(np.uint16)
        return Image.fromarray(np.add(np.random.randn(*np.shape(image_data)) * noise_magnitude, image_data).clip(0, 255).astype(np.uint8))
    f.num = num_images_total
    return f

def identity():
    def f(image, _):
        return image
    f.num = 1
    return f
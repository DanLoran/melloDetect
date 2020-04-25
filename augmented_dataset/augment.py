from PIL import Image, ImageOps
import numpy as np
from math import log
import os
import argparse

def rotator(image, num_images_total):
    rotation_amount = 360 / num_images_total
    next_rotation = 0
    while next_rotation < 360:
        yield image.rotate(next_rotation)
        next_rotation += rotation_amount

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
def white_balancer(image, num_images_total, temp_interval = [1900, 15000]):
    for offset in np.linspace(temp_interval[0], temp_interval[1], num=num_images_total):
        image_data = np.array(image.convert('RGB')).astype(np.uint16)
        channel_offsets = color_temp(offset)
        transformed_image_data = np.moveaxis(np.array(
                [image_data[:,:,0] * channel_offsets[0],
                 image_data[:,:,1] * channel_offsets[1],
                 image_data[:,:,2] * channel_offsets[2]]).clip(0, 255).astype(np.uint8), 0, 2)
        yield Image.fromarray(transformed_image_data)

def mirror(image):
    yield ImageOps.flip(image)
    yield ImageOps.mirror(image)

def noise(image, num_images_total, noise_magnitude):
    np.random.seed(123)
    for i in range(num_images_total):
        image_data = np.array(image.convert('RGB')).astype(np.uint16)
        yield Image.fromarray(np.add(np.random.randn(*np.shape(image_data)) * noise_magnitude, image_data).clip(0, 255).astype(np.uint8))

def augment(image):
    yield from rotator(image, 4)
    yield from white_balancer(image, 4)
    yield from mirror(image)
    yield from noise(image, 1, 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment an image, fool.')
    parser.add_argument('source_image', type=str, help='where to read the image')
    parser.add_argument('dest_path', type=str, help='the directory to write the image to')
    options = parser.parse_args()

    for idx, image in enumerate(augment(Image.open(options.source_image))):
        image.save(options.dest_path + os.path.splitext(os.path.basename(options.source_image))[0] + "_" + str(idx) + ".jpg")

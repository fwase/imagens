import cv2
import numpy as np
import json
from collections import Counter, defaultdict
from matplotlib import pyplot as plt

def read_img_rgb(img_file):
    img = cv2.imread(img_file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb


def show_img(img):
    plt.imshow(img)
    plt.show()


def increase_brightness(img, value=10):
    channels = cv2.split(img)

    for i in range(len(channels)):
        height, width = channels[i].shape
        
        for j in range(height):
            for k in range(width):
                if channels[i][j][k] + value <= 255:
                    channels[i][j][k] += value
                else:
                    channels[i][j][k] = 255

    final_img = cv2.merge(channels)

    return final_img


def negative(img):
    channels = cv2.split(img)

    for i in range(len(channels)):
        height, width = channels[i].shape
        
        for j in range(height):
            for k in range(width):
                channels[i][j][k] = 255 - channels[i][j][k]

    final_img = cv2.merge(channels)

    return final_img


def global_histogram(img, generate_array=True):
    if generate_array:
        channels = cv2.split(img)
        channels_pixel_values = [channel.flatten() for channel in channels]
        pixel_intensity = defaultdict(list)

        for channel_pixel_value in channels_pixel_values:
            pixel_frequency = Counter(channel_pixel_value)
            
            for key, item in pixel_frequency.items():
                pixel_intensity[key].append(item)

        pixel_intensity_sorted = sorted(pixel_intensity.items(), key=lambda k: k[0])
        pixel_intensity_sorted = {int(item[0]): item[1] for item in pixel_intensity_sorted}
        with open('represent_histogram_of_image.json', 'w') as f:
            json.dump(dict(pixel_intensity_sorted), f)

    for i, col in enumerate(['red', 'green', 'blue']):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.figure(figsize=(15,6))
        plt.title(f'Channel {i}', fontsize=15)
        plt.xlabel('Pixel intensity', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.plot(hist, color=col)
        plt.xlim([0,256])

    plt.show()

def compression_and_expansion(img):
    def map_pixel(pixel):
        if pixel <= 85:
            return pixel // 2
        
        if 85 < pixel < 170:
            return int(2 * pixel - 127)
        
        return int(pixel / 2 + 128)


    channels = cv2.split(img)

    for i in range(len(channels)):
        height, width = channels[i].shape

        for j in range(height):
            for k in range(width):
                channels[i][j][k] = map_pixel(channels[i][j][k])

    final_img = cv2.merge(channels)

    return final_img


def linear_contrast_expansion(img, z_a, z_b, min_value = 0, max_value = 255):
    def map_pixel(pixel, z_a, z_b, min_value, max_value):
        if pixel <= z_a:
            return min_value

        if z_a < pixel < z_b:
            return int(
                ((max_value - min_value) / (z_b - z_a)) * (pixel - z_a) + min_value
            )

        return max_value

    channels = cv2.split(img)

    for i in range(len(channels)):
        height, width = channels[i].shape

        for j in range(height):
            for k in range(width):
                channels[i][j][k] = map_pixel(channels[i][j][k], z_a, z_b, min_value, max_value)

    final_img = cv2.merge(channels)
    
    return final_img    
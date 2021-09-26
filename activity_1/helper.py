import json
import os
import statistics
from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans


def read_img(img_file, gray_scale=False):
    img = cv2.imread(img_file)

    if gray_scale:
        return cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


def counter(array):
    frequency = {}

    for i in range(256):
        frequency[i] = 0

    for value in array:
        frequency[value] += 1

    return frequency


def count_pixels_frequency(img):
    channels = cv2.split(img)
    channels_pixel_values = [channel.flatten() for channel in channels]
    pixel_freq = defaultdict(list)

    for channel_pixel_value in channels_pixel_values:
        pixel_frequency = counter(channel_pixel_value)

        for key, item in pixel_frequency.items():
            pixel_freq[key].append(item)

    return pixel_freq


def global_histogram(img, path_file_output):
    pixels_freq = count_pixels_frequency(img)

    with open(path_file_output, "w") as f:
        json.dump(pixels_freq, f)

    print("Done!")
    print(f"Saved in {path_file_output}")


def break_in_blocks(img, grid_order):
    (height, width) = img.shape[:2]
    blocks = []

    for i in range(grid_order):
        for j in range(grid_order):
            img_grid = img[
                i * height // grid_order : i * height // grid_order
                + height // grid_order,
                j * width // grid_order : j * width // grid_order + width // grid_order,
                :,
            ]
            blocks.append(img_grid)

    return blocks


def local_histogram(img, grid_order, path=""):
    blocks = break_in_blocks(img, grid_order)
    path_file = os.path.join(path, "represent_local_histogram_of_image.json")
    pixels_freq_block = [count_pixels_frequency(block) for block in blocks]

    with open(path_file, "w") as f:
        for pixel_freq_block in pixels_freq_block:
            json.dump(pixel_freq_block, f)
            f.write("\n")

    print("Done!")
    print(f"Saved in {path_file}")


def compression_and_expansion(img):
    def map_pixel(pixel):
        if pixel <= 85:
            return pixel // 2

        if 85 < pixel < 170:
            return 2 * pixel - 127

        return pixel // 2 + 128

    channels = cv2.split(img)

    for i in range(len(channels)):
        height, width = channels[i].shape

        for j in range(height):
            for k in range(width):
                channels[i][j][k] = map_pixel(channels[i][j][k])

    final_img = cv2.merge(channels)

    return final_img


def linear_contrast_expansion(img, z_a, z_b, min_value=0, max_value=255):
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
                channels[i][j][k] = map_pixel(
                    channels[i][j][k], z_a, z_b, min_value, max_value
                )

    final_img = cv2.merge(channels)

    return final_img


def histogram(img, title):
    height, width, shape = img.shape
    histogramArray = np.repeat(0, 256)

    for i in range(0, width):
        for j in range(0, height):
            histogramArray[img[j][i]] += 1

    if title:
        plt.title(title, fontsize=15)
        plt.xlabel("Intensidade do pixel", fontsize=10)
        plt.ylabel("Frequência", fontsize=10)
        plt.plot(histogramArray)
        plt.show()
        plt.clf()

    return histogramArray


def partition_histogram(img, path_file_output, generate_array=True):
    height, width, shape = img.shape

    part1 = img[0 : int(height / 2), 0 : int(width / 2)]
    part2 = img[0 : int(height / 2), int(width / 2) : width]
    part3 = img[int(height / 2) : height, 0:width]

    show_img(part1)
    partition_1_histogram = histogram(part1, "Partição superior esquerda")

    show_img(part2)
    partition_2_histogram = histogram(part2, "Partição superior direita")

    show_img(part3)
    partition_3_histogram = histogram(part3, "Partição inferior")

    if generate_array:
        partitions = [
            partition_1_histogram,
            partition_2_histogram,
            partition_3_histogram,
        ]

        with open(path_file_output, "w") as f:
            for partition in partitions:
                partition_dict = {i: int(partition[i]) for i in range(len(partition))}
                json.dump(partition_dict, f)
                f.write("\n")

        print("Done!")
        print(f"Saved in {path_file_output}")


def mean_filter(img):
    imgWithDuplicatedEdge = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    height, width = imgWithDuplicatedEdge.shape[:2]

    meanFilterOutput = img

    for row in range(1, height - 2):
        for column in range(1, width - 2):
            pixel_matrix = imgWithDuplicatedEdge[
                row - 1 : row + 2, column - 1 : column + 2
            ]

            b_mean = int(np.sum(pixel_matrix[:, :, 0]) / 9)
            r_mean = int(np.sum(pixel_matrix[:, :, 1]) / 9)
            g_mean = int(np.sum(pixel_matrix[:, :, 2]) / 9)

            meanFilterOutput[row][column] = [b_mean, r_mean, g_mean]

    return meanFilterOutput


def mode_filter(img, matrizSize=3):
    imgWithDuplicatedEdge = cv2.copyMakeBorder(
        img, matrizSize, matrizSize, matrizSize, matrizSize, cv2.BORDER_REFLECT
    )
    height, width = imgWithDuplicatedEdge.shape[:2]

    outputImage = img
    halfMatrizSize = int(matrizSize / 2)

    for row in range(matrizSize, height - matrizSize * 2):
        for column in range(matrizSize, width - matrizSize * 2):
            pixel_matrix = imgWithDuplicatedEdge[
                row - halfMatrizSize : row + halfMatrizSize + 1,
                column - halfMatrizSize : column + halfMatrizSize + 1,
            ]

            b_mode = statistics.mode(pixel_matrix[:, :, 0].flatten())
            r_mode = statistics.mode(pixel_matrix[:, :, 1].flatten())
            g_mode = statistics.mode(pixel_matrix[:, :, 2].flatten())

            outputImage[row][column] = [b_mode, r_mode, g_mode]

    return outputImage


def quantization(img, total_colors):
    (height, width) = img.shape[:2]
    image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    k_means = MiniBatchKMeans(n_clusters=total_colors)
    labels = k_means.fit_predict(image)
    quant = k_means.cluster_centers_.astype("uint8")[labels]

    image_quant = quant.reshape((height, width, 3))
    image_quant = cv2.cvtColor(image_quant, cv2.COLOR_LAB2RGB)

    return image_quant

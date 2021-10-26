import cv2
from matplotlib import pyplot as plt
from numpy import dot
from numpy.linalg import norm


def read_img(img_file, gray_scale=False):
    img = cv2.imread(img_file)

    if gray_scale:
        return cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_img(img):
    plt.imshow(img)
    plt.show()


def histogram(img):
    total_pixels = 256
    histograms = {}

    for index, channel in enumerate(["red", "green", "blue"]):
        counter = cv2.calcHist([img], [index], None, [total_pixels], [0, total_pixels])
        histograms[channel] = counter.reshape(1, total_pixels)[0]

    return histograms


def cossine_similarity(array_1, array_2):
    return dot(array_1, array_2) / (norm(array_1) * norm(array_2))


def distance_two_frames(frame_1, frame_2):
    channels = ["red", "green", "blue"]
    histograms_frame_1 = histogram(frame_1)
    histograms_frame_2 = histogram(frame_2)

    distance = {}

    for channel in channels:
        hist_frame_1 = histograms_frame_1.get(channel)
        hist_frame_2 = histograms_frame_2.get(channel)
        distance[channel] = cossine_similarity(hist_frame_1, hist_frame_2)

    return distance


def get_total_frames(video):
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))


def get_fps(video):
    return video.get(cv2.CAP_PROP_FPS)


def get_seconds_from_frame_number(frame_number, fps):
    return int(frame_number / fps)


def get_frames_from_video(video, window):
    total_frames = get_total_frames(video)
    frames = []

    for index_frame in range(total_frames):
        _, frame = video.read()
        if index_frame % window == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    return frames


def change_scene(frame_1, frame_2, threshold):
    distance = distance_two_frames(frame_1, frame_2)
    min_distance = min(distance.values())

    return True if min_distance < threshold else False


def accuracy(change_scenes_true, change_scenes_predict):
    total = len(change_scenes_true)
    correct = 0

    for second in change_scenes_true:
        if second in change_scenes_predict:
            correct += 1

    return correct / total


def generate_index_frames_to_compare(video, window):
    total_frames = get_total_frames(video)
    frames_to_compare = []

    for index_frame in range(0, total_frames - window, window):
        frames_to_compare.append((index_frame, index_frame + window))

    return frames_to_compare


def divide_3_partitions(image):
    height, width, _ = image.shape

    part_1 = image[0 : int(height / 2), 0 : int(width / 2)]
    part_2 = image[0 : int(height / 2), int(width / 2) : width]
    part_3 = image[int(height / 2) : height, 0:width]

    return [part_1, part_2, part_3]


def divide_5_by_5(image):
    partitions = []
    height, width, _ = image.shape
    height = int(height / 5)
    width = int(width / 5)

    for i in range(5):
        for j in range(5):
            partitions.append(
                image[i * height : (i + 1) * height, j * width : (j + 1) * width]
            )

    return partitions


def save_image(image, path_file):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path_file, image_bgr)

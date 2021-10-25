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


def get_frames_from_video(video):
    total_frames = get_total_frames(video)
    frames = []

    for _ in range(total_frames):
        _, frame = video.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    return frames

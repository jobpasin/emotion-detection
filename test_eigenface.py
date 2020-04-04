import os, sys
import numpy as np
import sklearn
from PIL import Image
from utils.eigenface import *
from sklearn.metrics import precision_score


def detect_emotion(test_image_folder, eigenface_path):
    """
    Test images and get the emotion
    :param test_image_folder:
    :param eigenface_path:
    :return:
    """
    all_image_path = import_image_path(test_image_folder)
    all_image_path.sort()
    eigen_data = load_eigenface(eigenface_path)
    eigen_data['loss'] = []

    test_image_path = []
    predict_image_emotion = []
    ground_truth_emotion = []
    try:
        for p in all_image_path:
            ground_truth_emotion.append(eigen_data['emotion_name'].index(os.path.basename(p).split('_')[0]))
            test_image_path.append(p)
    except ValueError:
        print("Warning: {} is not in list. Full filename: {}".format(os.path.basename(p).split('_')[0], p))

    for im_path in test_image_path:
        test_data = load_image(im_path)
        data_loss = []
        for em_index, emotion in enumerate(eigen_data['emotion_name']):
            em_eigenface = eigen_data['eigenface'][em_index]
            em_image_average = eigen_data['image_average'][em_index]
            em_loss = reconstruction_loss(test_data, em_eigenface, em_image_average)
            data_loss.append(em_loss)

        # Choose the index with minimum loss
        predict_image_emotion.append(data_loss.index(min(data_loss)))

    return ground_truth_emotion, predict_image_emotion


def get_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    return precision


def load_eigenface(eigenface_path):
    data = np.load(eigenface_path, mmap_mode='r')
    eigenface_data = {'emotion_name': ['hap', 'sad', 'ang', 'sur'],
                      'eigenface': [data.f.Eig0, data.f.Eig1, data.f.Eig2, data.f.Eig3],
                      'image_average': [data.f.ImAv0, data.f.ImAv1, data.f.ImAv2, data.f.ImAv3]}
    return eigenface_data


if __name__ == "__main__":
    test_folder = "./TestingPics"
    eigenface_path = "./weights/train_debug.npz"

    ground_truth, prediction = detect_emotion(test_folder, eigenface_path)
    precision = get_metrics(ground_truth, prediction)
    print(precision)

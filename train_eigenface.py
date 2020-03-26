# Training Process
# Reference: https://www.mitpressjournals.org/doi/10.1162/jocn.1991.3.1.71

# !/usr/bin/python
import os

# original_dir = os.getcwd()
# sys.path.append('/home/pi/.virtualenvs/cv/lib/python3.6/site-packages')  # For Raspberry Pi
import timeit
import numpy as np
from PIL import Image
from utils.eigenface import create_eigface


def import_images(paths, emotion, img_indexes=None, img_amount=None):
    """
    Fetch list of images directory of specific emotion. Specifically work for our file format.
    :param paths:       String, folder directory where images are stored. Can be list to fetch from multiple folders.
    :param emotion:     String of emotion name. Image name should start with emotion e.g. sad_1010.jpg
    :param img_indexes: List of string of specific image id you want e.g. [1010.jpg]. If None, will fetch all
    :param img_amount:  Amount of image you want to fetch. Use None to fetch all
    :return:            ndarray of size (h*w, img_amount) or (features, num_image)
    """
    if type(paths) is not list:
        paths = [paths]

    image_paths = []
    for p in paths:
        if img_indexes is None or len(img_indexes) == 0:
            im_p = [os.path.join(p, f) for f in os.listdir(p) \
                    if f.split("_")[0].endswith(emotion)]  # Get path of file that start with 'emotion'
        else:
            im_p = [os.path.join(p, f) for f in os.listdir(p) \
                    if (f.split("_")[0].endswith(emotion) and f.split("_")[1] in img_indexes)]
        image_paths = image_paths + im_p
    image_paths.sort()

    if img_amount is None:
        img_amount = len(image_paths)
    else:
        assert img_amount <= len(image_paths), "Requested too many images. Amount of image found is {}.".format(
            len(image_paths))

    # Load all image and subtract the value by the average value
    image_vector = np.stack([load_image(image_path) for image_path in image_paths[0:img_amount]])
    image_average = np.tile(np.mean(image_vector, 0), (img_amount, 1)) / img_amount

    image_vector = image_vector - image_average
    # return the images list and average
    return image_vector.T, image_average.T


def load_image(full_path):  # Importing single image
    image = np.array(Image.open(full_path).convert('L'), 'float32')
    [h, w] = np.shape(image)
    image_vec = np.reshape(image, h * w, 'C')
    return image_vec


if __name__ == '__main__':
    tic = timeit.default_timer()

    # Parameters
    precent_eig_param = [90.0, 90.0, 90.0, 90.0]  # Minimum of percentage of eigenvalue
    num_eig_param = None  # Number of Eig
    train_dir = './TrainingPicsAll'  # Training pictures path
    img_index = []
    # img_index = ['1008.jpg', '1011.jpg', '1014.jpg', '1015.jpg', '1022.jpg', '1025.jpg',
    #              '1027.jpg', '1028.jpg', '1031.jpg', '1032.jpg', '1033.jpg', '1117.jpg',
    #              '1118.jpg', '1123.jpg', '1126.jpg', '1129.jpg']
    PCA_Path = './weights/train_debug.npz'  # Path to save PCA


    data = {'image_emotion': ['hap', 'sad', 'ang', 'sur'],
            'train_image': [], 'average_image': [],
            'eigenface.py': [], 'eigenvalue': []
            }

    for i, emo in enumerate(data['image_emotion']):
        train_image, im_avg = import_images(train_dir, emo, img_index)  # Importing image
        eigface, eigval = create_eigface(train_image, num_eigen=None, percent_eigen=precent_eig_param[i])  # Calculate eigenface
        data['train_image'].append(train_image)
        data['average_image'].append(im_avg)
        data['eigenface.py'].append(eigface)
        data['eigenvalue'].append(eigval)

    # Save to "PCATrain.npz"
    File = open(PCA_Path, 'w+')
    File.close()
    np.savez(PCA_Path,
             Eig0=data['eigenface.py'][0], ImAv0=data['average_image'][0],
             Eig1=data['eigenface.py'][1], ImAv1=data['average_image'][1],
             Eig2=data['eigenface.py'][2], ImAv2=data['average_image'][2],
             Eig3=data['eigenface.py'][3], ImAv3=data['average_image'][3])

    toc = timeit.default_timer()

    print("Total time spend for training is: {} seconds".format(toc - tic))
    print("Total images for training is {}, {}, {}, {} images (For each emotion)".format(
        np.shape(data['train_image'][0])[1], np.shape(data['train_image'][1])[1],
        np.shape(data['train_image'][2])[1], np.shape(data['train_image'][3])[1]))
    print("Image size is {} features".format(np.shape(data['train_image'][0])[0]))
    print("Number of optimal projection axis is:", np.shape(data['eigenface.py'][0])[1])

    # TODO: Add validation part in here instead of 1DPCA_Validation
    # TODO: Fix test part
'''
# Perform the tranining
recognizer.train(images, np.array(labels))
i = 0


# Append the images with the extension .sad into image_paths
image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
for image_path in image_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        i = i+1
        cropped = np.array(cv2.resize(predict_image[y: y + h, x: x + w],(150,150)),'uint8')
        nbr_predicted, conf = recognizer.predict(cropped)
        nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        if nbr_actual == nbr_predicted:
            print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
        else:
            print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
        cv2.imshow("Recognizing Face", cropped)
        cv2.waitKey(1000)
        
'''

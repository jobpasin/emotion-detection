# Training Process

# !/usr/bin/python
import os, sys

# original_dir = os.getcwd()
# sys.path.append('/home/pi/.virtualenvs/cv/lib/python3.6/site-packages')  # For Raspberry Pi
import timeit
import cv2
import numpy as np
from PIL import Image

# sys.path.append(original_dir)

# -------------------Initialize ----------------------------------
percent_Eig = [80.0, 80.0, 80.0, 80.0]  # Minimum of percentage of eigenvalue
num_Pic = 5  # Number of Pictures
image_dir = './TrainingPicsAll'  # Training pictures path
img_index = []
# img_index = ['1008.jpg', '1011.jpg', '1014.jpg', '1015.jpg', '1022.jpg', '1025.jpg',
#              '1027.jpg', '1028.jpg', '1031.jpg', '1032.jpg', '1033.jpg', '1117.jpg',
#              '1118.jpg', '1123.jpg', '1126.jpg', '1129.jpg']
PCA_Path = './TrainedData/new_test_6.npz'  # Path to save PCA

# ----------------------------------------------------------------

tic = timeit.default_timer()


def import_images(path, num_pic, emotion, img_indexes=None):
    """
    Fetch list of images directory of specific emotion. Specifically work for our file format.
    :param path:        Folder directory where images are stored. Can be list to fetch from multiple folders.
    :param num_pic:
    :param emotion:     String of emotion name. Image name should start with emotion e.g. sad_1010.jpg
    :param img_indexes: List of string of specific image id you want e.g. [1010.jpg]. Use None to fetch all
    """
    if type(path) is not list:
        path = [path]

    image_paths = []
    for p in path:
        if img_indexes is None:
            im_p = [os.path.join(p, f) for f in os.listdir(p) \
                    if f.split("_")[0].endswith(emotion)]  # Get path of file that start with 'emotion'
        else:
            im_p = [os.path.join(p, f) for f in os.listdir(p) \
                    if (f.split("_")[0].endswith(emotion) and f.split("_")[1] in img_indexes)]
        image_paths = image_paths + im_p
    image_paths.sort()

    # Fetch image size
    image_pil = Image.open(image_paths[0]).convert('L')
    [h, w] = np.shape(np.array(image_pil, 'float32'))

    # num_pic = len(image_paths)
    im_av = np.array(np.zeros([h * w, 1], 'float32'))  # Create average value
    # Label for each images
    for i,image_path in enumerate(image_paths):
        # Read the image and convert to grayscale, convert to matrix
        image_pil = Image.open(image_path).convert('L')
        # print image_path
        image = np.array(image_pil, 'float32')
        image_vec_temp = np.reshape(image, h * w, 'C').transpose()
        if image_path == image_paths[0]:
            image_vec = image_vec_temp
        else:
            image_vec = np.concatenate((image_vec, image_vec_temp), axis=1)
        im_av = im_av + image_vec_temp
        if i == num_pic:
            break
    im_av = im_av / num_pic
    image_vec = image_vec - np.tile(im_av, (1, num_pic))
    # return the images list and average
    return image_vec, im_av


def import_image(fullpath):  # Importing single image
    image = np.array(Image.open(fullpath).convert('L'), 'float32')
    [h, w] = np.shape(image)
    image_vec = np.reshape(image, h * w, 'C').transpose()
    return image_vec


def create_eigface(images, im_av, percent_Eig):  # Process of creating optimal projection axis

    [Size, num_pic] = images.shape  # Number of training images
    # Create pseudo covariance matrix
    Cov = images.transpose() * images

    # Compute Eigenvalue and Eigenvector
    [EgVal, EgVec_temp] = np.linalg.eig(Cov)
    EgVec_temp = np.matrix(EgVec_temp)
    EgVec = images * EgVec_temp
    index = EgVal.argsort()[::-1]
    EgVal = EgVal[index]
    EgVec = EgVec[:, index]
    # Calculate number of eigenfaces
    Sum_EgVal = np.sum(EgVal, dtype='float32')
    EgVal_percent = EgVal / Sum_EgVal * 100
    num_Eig = 1
    for i in range(1, num_Pic):
        if np.sum(EgVal_percent[0:i]) > percent_Eig:
            break
        num_Eig = num_Eig + 1
    # Select only num_Eig vectors and normalize to size of 1
    Eig = EgVec[:, 0] / np.linalg.norm(EgVec[:, 0])
    for cnt in range(1, num_Eig):
        Eig = np.concatenate((Eig, EgVec[:, cnt] / np.linalg.norm(EgVec[:, cnt])), axis=1)
    # return optimal projection axis and average image for future use
    return Eig, EgVal


def reconstruct_image(Eig, num_Eig, Mode):  # Return eigenface into image
    # ReconIm(Eig[0],0,'E') or RecomIm(Im_Av[0],0,'Av')

    if Mode == "E":
        Eig_Im = np.reshape(Eig[:, num_Eig], (100, 100))
        Eig_Im = Eig_Im - np.matrix.min(Eig_Im)
        Eig_Im = np.array(Eig_Im / np.max(Eig_Im) * 255, dtype=np.uint8)
    elif Mode == "E2":
        Eig_Im = np.reshape(Eig[:, num_Eig], (100, 100))
        Eig_Im = -Eig_Im
        Eig_Im = Eig_Im - np.matrix.min(Eig_Im)
        Eig_Im = np.array(Eig_Im / np.max(Eig_Im) * 255, dtype=np.uint8)
    elif Mode == "Av":
        Eig_Im = np.reshape(Eig, (100, 100))
        Eig_Im = np.array(Eig_Im, dtype=np.uint8)
    else:
        print("Error mode : 'E' or 'Av'")
    Eig_Im = cv2.resize(Eig_Im, (400, 400))
    cv2.imshow("Eigenface Image: " + str(num_Eig), Eig_Im)

    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return


# ----------------Main Code -----------------------------#
if __name__ == '__main__':
    # Importing image
    train_images = range(4)
    Im_Av = range(4)
    train_images[0], Im_Av[0] = import_images(image_dir, num_Pic, 'hap', img_index)
    train_images[1], Im_Av[1] = import_images(image_dir, num_Pic, 'sad', img_index)
    train_images[2], Im_Av[2] = import_images(image_dir, num_Pic, 'ang', img_index)
    train_images[3], Im_Av[3] = import_images(image_dir, num_Pic, 'sur', img_index)
    # train_images[cnt],Im_Av[cnt] = import_images(path,'happy')
    # train_images[cnt],Im_Av[cnt] = import_images(path,'sad')
    # train_images[cnt],Im_Av[cnt] = import_images(path,'normal')
    # train_images[cnt],Im_Av[cnt] = import_images(path,'surprised')

    # Create Eigenface
    Eig = range(4)
    Eig_Val = range(4)
    for cnt in range(0, 4):
        Eig[cnt], Eig_Val[cnt] = create_eigface(train_images[cnt], Im_Av[cnt], percent_Eig[cnt])

    # Save to "PCATrain.npz"
    File = open(PCA_Path, 'w+')
    File.close()
    np.savez(PCA_Path, Eig0=Eig[0], ImAv0=Im_Av[0], Eig1=Eig[1], ImAv1=Im_Av[1] \
             , Eig2=Eig[2], ImAv2=Im_Av[2], Eig3=Eig[3], ImAv3=Im_Av[3])

    toc = timeit.default_timer()

    print("Total time spend for training is:", toc - tic, "seconds")
    print("Total images for training is", np.shape(train_images[0])[1], ",", np.shape(train_images[1])[1], ",", \
          np.shape(train_images[2])[1], ",", np.shape(train_images[3])[1], "images (For each emotion)")
    print("Image size is", train_images[0][0].shape)
    print("Number of optimal projection axis is:", len(Eig[0]))

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

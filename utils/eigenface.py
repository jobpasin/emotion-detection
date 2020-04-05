import cv2
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import normalize

def create_eigface(images, num_eigen=None, percent_eigen=None):  # Process of creating optimal projection axis
    """
    Function to create eigenface.py (Using Principal Component Analysis with some extra technique to save computation cost)
    :param images           : ndarray of size (features, num_image), (N^2, M)
    :param num_eigen        : int, number of eigenfaces. Cannot be specified with percent_eigen
    :param percent_eigen    : float, number of percentage of eigenvalues for determine number of eigenface.py

    """
    features, num_pic = np.shape(images)
    if num_eigen is not None and percent_eigen is not None:
        raise ValueError("Only one of 'num_eigen' or 'percent_eigen' can be specified")
    if num_eigen is None and percent_eigen is None:
        num_eigen = num_pic  # Use number of eigenvector as number of images

    # Create pseudo covariance matrix. This case will have matrix size [num_pic, num_pic]
    # which can be much smaller than standard approach where cov matrix will be [features, features]
    cov = np.matmul(images.transpose(), images)  # (M,M)

    # Compute Eigenvalue and Eigenvector
    eig_value, eig_vector_temp = np.linalg.eig(cov)  # eigenvector v_i (M features)

    # Find true eigenvector
    eig_vector = np.matmul(images, eig_vector_temp)  # Convert to u_i = A*v_i (N^2 features)
    eig_vector = normalize(eig_vector, norm='l2', axis=0)  # Normalize so |u| = 1

    # Sort eigenvalue and correspond eigenvector in descending order
    index = eig_value.argsort()[::-1]
    eig_value = eig_value[index]
    eig_vector = eig_vector[:, index]

    if percent_eigen is not None:  # Select number of eigenvector based on percentage of eigenvalue
        assert 0 < percent_eigen < 100, "percent_eigen need to be between 0 to 100"
        # Calculate amount of eigenfaces by select the minimum amount that has total eigenvalues higher than threshold
        sum_eigenvalue = np.sum(eig_value, dtype='float32')
        for i in range(1, num_pic):
            if np.sum(eig_value[0:i]) > percent_eigen * sum_eigenvalue / 100:
                num_eigen = i
                print("Using {} eigenfaces".format(num_eigen))
                break
    else:  # Select number of eigenvector based on input directly
        if num_pic < num_eigen:
            num_eigen = num_pic
            print("Warning: Number of eigenfaces is limited to no more than the number of training images")

    # Fetch a selected number of eigenface.py and eigenvalue
    eigenface = eig_vector[:, 0:num_eigen]

    # return optimal projection axis and average image for future use
    return eigenface, eig_value


def preprocess_test(test_image, image_average):
    return test_image - image_average


def reconstruction_loss(test_im, eigenface, im_average):  # Process of testing
    # Number of eigenfaces, normalized image
    # test_image = np.reshape(test_image, np.shape(test_image)[0] * np.shape(test_image)[1], 'C').transpose()  # Vectorize
    test_im = preprocess_test(test_im, im_average)  # Normalize with average of training data

    # Create projection face from training eigenface.py
    image_weight = np.matmul(eigenface.transpose(), test_im)
    projected_image = np.matmul(eigenface, image_weight)  # Principle component of image using optimal projection

    # Calculate total euclidean distance
    error = np.linalg.norm(test_im - projected_image)
    return error


def visualize_eigenface(eigenface, eigenface_index, mode="default", image_dim=[100, 100]):
    """
    Transform the eigenface into an image and display with openCV for visualization
    :param images           : ndarray of image or eigenface with dimension
    :param eigenface_index    : int, the index of eigenfaces used to visualize
    :param mode             : string. Type of image of the following option.
                            :'default' will visualize eigenface of size [features, num_eigenface]
                            :'inverse' will give the inverted image instead
                            :'raw' for vectorized image of size [features]
    """
    # Return eigenface.py into image
    # reconstruct_image(eigenface[0],0,'eig') or reconstruct_image(eig_value[0],0,'normal_im')
    mode = mode.lower()
    mode_option = ['default', 'inverse', 'raw']
    assert mode in mode_option, "Mode need to be one of the following: {}".format(mode_option)
    if mode == "raw":
        eigen_image = np.reshape(eigenface, (image_dim[0], image_dim[1]))
        eigen_image = np.array(eigen_image, dtype=np.uint8)
    else:
        eigen_image = np.reshape(eigenface[:, eigenface_index], (image_dim[0], image_dim[1]))
        if mode == "inverse":
            eigen_image = -eigen_image
        eigen_image = eigen_image - np.matrix.min(eigen_image)
        eigen_image = np.array(eigen_image / np.max(eigen_image) * 255, dtype=np.uint8)

    im = cv2.resize(eigen_image, (400, 400))
    cv2.imshow("Eigenface Image: " + str(eigenface_index), im)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return


def import_image_path(folder_dir):
    """
    Fetch path of all file in the folder directory
    :param path: String folder path
    :param img_indexes: 
    :return: 
    """""
    return [os.path.join(folder_dir, f) for f in os.listdir(folder_dir)]


def load_image(full_path):  # Importing single image
    image = np.array(Image.open(full_path).convert('L'), 'float32')
    [h, w] = np.shape(image)
    image_vec = np.reshape(image, h * w, 'C')
    return image_vec


def preprocess_train(image_path_list):
    # Load all image and subtract the value by the average value
    img_amount = len(image_path_list)
    image_vector = np.stack([load_image(image_path) for image_path in image_path_list])
    image_average = np.mean(image_vector, 0)

    image_vector = image_vector - np.tile(image_average, (img_amount, 1))
    # return the images list and average
    return np.transpose(image_vector), np.transpose(image_average)


if __name__ == "__main__":
    image = np.array([[1, 2, 3], [4, 5, 6], [9, 8, 7], [12, 11, 10]])
    image_average = np.mean(image, 0)
    image = image - np.tile(image_average, (4, 1))
    print("Image of {} features and {} images".format(np.shape(image)[0], np.shape(image)[1]))
    eigenface, eigenvalue = create_eigface(image)
    print("Finish)")

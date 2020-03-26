import cv2
import numpy as np


def create_eigface(images, num_eigen=None, percent_eigen=None):  # Process of creating optimal projection axis
    """
    Function to create eigenface.py (Using Principal Component Analysis with some extra technique to save computation cost)
    :param images           : ndarray of size (features, num_image)
    :param num_eigen        : int, number of eigenfaces. Cannot be specified with percent_eigen
    :param percent_eigen    : float, number of percentage of eigenvalues for determine number of eigenface.py

    """
    if num_eigen is not None and percent_eigen is not None:
        raise ValueError("Only one of 'num_eigen' or 'percent_eigen' can be specified")
    if num_eigen is None and percent_eigen is None:
        raise ValueError("Either 'num_eigen' or 'percent_eigen' need to be specified")

    features, num_pic = np.shape(images)
    # Create pseudo covariance matrix. This case will have matrix size [num_pic, num_pic]
    # which can be much smaller than standard approach where cov matrix will be [features, features]
    cov = np.matmul(images.transpose(), images)

    # Compute Eigenvalue and Eigenvector
    eig_value, eig_vector_temp = np.linalg.eig(cov)

    # Find true eigenvector
    eig_vector = np.matmul(images, eig_vector_temp)  # Convert vector back since we do transpose of normal cov matrix

    # Sort eigenvalue and correspond eigenvector by descending
    index = eig_value.argsort()[::-1]
    eig_value = eig_value[index]
    eig_vector = eig_vector[:, index]

    if percent_eigen is not None:  # percent_eigen mode
        assert 0 < percent_eigen < 100, "percent_eigen need to be between 0 to 100"
        # Calculate amount of eigenfaces by select the minimum amount that has total eigenvalues higher than threshold
        sum_eigenvalue = np.sum(eig_value, dtype='float32')
        for i in range(1, num_pic):
            if np.sum(eig_value[0:i]) > percent_eigen * sum_eigenvalue / 100:
                num_eigen = i
                print("Using {} eigenfaces".format(num_eigen))
                break
    else:  # num_eigen mode
        if num_pic < num_eigen:
            num_eigen = num_pic
            print("Warning: Eigenfaces is limited to no more than the number of training images")

    # Fetch a selected number of eigenface.py and eigenvalue
    eigenface = eig_vector[:, 0:num_eigen]

    # Select only num_Eig vectors and normalize to size of 1
    # eigenface.py = eig_vector/eig_vector.sum(axis=0,keepdims=1)  # Unused

    # return optimal projection axis and average image for future use
    return eigenface, eig_value


def reconstruction_loss(test_image, eigenface, im_average):  # Process of testing
    # Number of eigenfaces, normalized image
    test_image = np.reshape(test_image, np.shape(test_image)[0] * np.shape(test_image)[1], 'C').transpose()  # Vectorize
    test_image = test_image - im_average  # Normalize with average of training data

    # Create projection face from training eigenface.py
    test_weight = eigenface.transpose() * test_image
    projected_image = eigenface * test_weight  # Principle component of image using optimal projection

    # Calculate total euclidean distance
    error = np.linalg.norm(test_image - projected_image)
    return error


def reconstruct_image(images, num_eigenface, mode, image_dim=[100, 100]):
    """
    Transform the eigenface.py into an image and display with openCV for visualization
    :param images           : ndarray of image or eigenface.py with dimension
    :param num_eigenface    : int, amount of eigenface.py to visualize
    :param mode             : string. Type of image of the following option. Case-insensitive.
                            :'eig' or 'eig_inv' for eigenface.py of size [features, num_eigenface]
                            :'eig_inv' will give the inverted image instead
                            :'normal_im' for vectorized image of size [features]
    """
    # Return eigenface.py into image
    # reconstruct_image(eigenface.py[0],0,'eig') or reconstruct_image(eig_value[0],0,'normal_im')
    mode = mode.lower()
    mode_option = ['eig', 'eig_inv', 'normal_im']
    assert mode in mode_option, "Mode need to be one of the following: {}".format(mode_option)
    if mode == "eig":
        eigen_image = np.reshape(images[:, num_eigenface], (image_dim[0], image_dim[1]))
        eigen_image = eigen_image - np.min(eigen_image)
        eigen_image = np.array(eigen_image / np.max(eigen_image) * 255, dtype=np.uint8)
    elif mode == "eig_inv":
        eigen_image = np.reshape(images[:, num_eigenface], (image_dim[0], image_dim[1]))
        eigen_image = -eigen_image
        eigen_image = eigen_image - np.matrix.min(eigen_image)
        eigen_image = np.array(eigen_image / np.max(eigen_image) * 255, dtype=np.uint8)
    elif mode == "normal_im":
        eigen_image = np.reshape(images, (image_dim[0], image_dim[1]))
        eigen_image = np.array(eigen_image, dtype=np.uint8)
    else:
        print("Error mode : 'E' or 'Av'")
    im = cv2.resize(eigen_image, (400, 400))
    cv2.imshow("Eigenface Image: " + str(num_eigenface), im)

    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return
# Facial Recognition Detection

## Goal & Algorithm
Our repository goals is to create a real-time emotion detection using machine learning algorithms. 
Our detection consists of two steps:
* Face detection using Haar features on openCV. (Currently not in the repo, will be updated soon) 
* Emotion detection based on eigenface algorithm. We create our own function in this step

## How to use (Eigenface Part)
1. Run `train_eigenface.py`. Put all training images in one folder and set the directory on parameter `train_dir`. After the training, the eigenface will be saved in `.weights/trained_eigenface.npz`. This is important for prediction.

**Note**: All image file must starts with 'hap', 'sad', 'ang', 'sur' follow by '\_' which correspond for happy, sadness, angry and surprised emotion. E.g. `hap_0001.jpg`

2. Run `test_eigenface.py`. Choose the folder to detect the emotion and the program will print out the emotion for each one. _(Currently improving the output to make it more readable)_

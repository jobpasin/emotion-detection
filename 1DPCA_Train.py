#Training Process

#!/usr/bin/python
import os, sys
Original_Dir = os.getcwd()
#sys.path.append('/home/pi/.virtualenvs/cv/lib/python2.7/site-packages') #####
import timeit
import cv2
import numpy as np
from PIL import Image
sys.path.append(Original_Dir)

#-------------------Initialize ----------------------------------
num_Eig = 16                                #Number of Eig
num_Pic = 21                                #Number of Pictures
path = './TrainingPics'                     #Training pictures path
path = './NitadDB/TrainingA'
path2 = ''
#path2 = './NitadInstantTraining/Haar'      #Additional training path
PCA_Path = './TrainedData/1DPCA_2_Test.npz'   #Path to save PCA


#----------------------------------------------------------------

tic = timeit.default_timer()

def import_images(path,path2, num_pic,emotion): #Process of importing images from a folder
    #Insert image path with the interested emotion
    #image_paths = [os.path.join(path, f) for f in os.listdir(path) \
                   #if f.split(".")[0].startswith(emotion)]
    image_paths = [os.path.join(path, f) for f in os.listdir(path) \
                   if f.split("_")[0].endswith(emotion)]
    if path2 != '':  
        image_paths2 = [os.path.join(path2, f) for f in os.listdir(path2) \
                       if f.split("_")[0].endswith(emotion)]
        image_paths = image_paths+image_paths2
    image_paths.sort()
    #image_paths = image_paths[::-1]
    # Array of image data
    image_pil = Image.open(image_paths[0]).convert('L')
    [H,W] = np.shape(np.array(image_pil,'float32'))
    
    #num_pic = len(image_paths)
    im_av = np.array(np.zeros([H*W,1],'float32')) #Create average value
    # Label for each images
    #labels = []
    i = 0;
    for image_path in image_paths:
        # Read the image and convert to grayscale, convert to matrix
        image_pil = Image.open(image_path).convert('L')
        #print image_path
        image = np.matrix(np.array(image_pil,'float32'))
        image_vec_temp = np.reshape(image,H*W,'C').transpose()
        if image_path == image_paths[0]:
            image_vec = image_vec_temp
        else:
            image_vec = np.concatenate((image_vec,image_vec_temp),axis=1)
        im_av = im_av+image_vec_temp
        i = i+1;
        if i== num_pic:
            break
    im_av = im_av/num_pic
    image_vec = image_vec - np.tile(im_av,(1,num_pic))
    # return the images list and average
    return image_vec, im_av

def import_image(fullpath): #Importing single image
    image = np.matrix(np.array(Image.open(fullpath).convert('L'),'float32'))
    [H,W] = np.shape(image)
    image_vec = np.reshape(image,H*W,'C').transpose()
    return image_vec

    
def create_Eig(images,im_av,num_Eig): #Process of creating optimal projection axis
    
    [Size,num_pic] = images.shape # Number of training images
    if num_pic < num_Eig:
        num_Eig = num_pic
        print "!!Eigenfaces is limited to number of training images"
    #Create pseudo covariance matrix
    Cov = images.transpose()*images

    #Compute Eigenvalue and Eigenvector
    [EgVal,EgVec_temp] = np.linalg.eig(Cov)
    EgVec_temp = np.matrix(EgVec_temp)
    EgVec = images*EgVec_temp
    index = EgVal.argsort()[::-1]
    EgVal = EgVal[index]
    EgVec_temp = EgVec[index]
    #Select only num_Eig vectors and normalize to size of 1
    Eig = EgVec[:,0]/np.linalg.norm(EgVec[:,0])
    for cnt in range(1,num_Eig):
        Eig = np.concatenate((Eig,EgVec[:,cnt]/np.linalg.norm(EgVec[:,cnt])),axis=1)

    # return optimal projection axis and average image for future use
    return Eig,EgVal

def ReconIm(Eig,num_Eig,Mode):  #Return eigenface into image
    #ReconIm(Eig[0],0,'E') or RecomIm(Im_Av[0],0,'Av')
    
    if Mode == "E":
        Eig_Im = np.reshape(Eig[:,num_Eig],(100,100))
        Eig_Im = Eig_Im-np.matrix.min(Eig_Im)
        Eig_Im = np.array(Eig_Im/np.max(Eig_Im)*255, dtype = np.uint8)
    elif Mode == "E2":
        Eig_Im = np.reshape(Eig[:,num_Eig],(100,100))
        Eig_Im = -Eig_Im
        Eig_Im = Eig_Im-np.matrix.min(Eig_Im)
        Eig_Im = np.array(Eig_Im/np.max(Eig_Im)*255, dtype = np.uint8)        
    elif Mode == "Av":
        Eig_Im = np.reshape(Eig,(100,100))
        Eig_Im = np.array(Eig_Im, dtype = np.uint8)
    else:
        print "Error mode : 'E' or 'Av'"
    Eig_Im = cv2.resize(Eig_Im,(400,400)) 
    cv2.imshow("Eigenface Image: "+str(num_Eig),Eig_Im)
    
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return
#----------------Main Code -----------------------------#

# Importing image
train_images = range(4)
Im_Av = range(4)
train_images[0],Im_Av[0] = import_images(path,path2,num_Pic,'hap')
train_images[1],Im_Av[1] = import_images(path,path2,num_Pic,'sad')
train_images[2],Im_Av[2] = import_images(path,path2,num_Pic,'ang') 
train_images[3],Im_Av[3] = import_images(path,path2,num_Pic,'sur')
#train_images[cnt],Im_Av[cnt] = import_images(path,'happy')
#train_images[cnt],Im_Av[cnt] = import_images(path,'sad')
#train_images[cnt],Im_Av[cnt] = import_images(path,'normal') 
#train_images[cnt],Im_Av[cnt] = import_images(path,'surprised')

# Create Eigenface 
Eig = range(4)
Eig_Val = range(4)
for cnt in range(0,4):
    Eig[cnt],Eig_Val[cnt] = create_Eig(train_images[cnt],Im_Av[cnt],num_Eig)
                                     
#Save to "PCATrain.npz"
File = open(PCA_Path,'w+')
File.close()
np.savez(PCA_Path,Eig0 = Eig[0],ImAv0 = Im_Av[0],Eig1 = Eig[1],ImAv1 = Im_Av[1] \
         ,Eig2 = Eig[2],ImAv2 = Im_Av[2],Eig3 = Eig[3],ImAv3 = Im_Av[3])       
                                                           
toc = timeit.default_timer()

print "Total time spend for training is:", toc-tic, "seconds"
print "Total images for training is", np.shape(train_images[0])[1], ",",np.shape(train_images[1])[1],",",\
      np.shape(train_images[2])[1],",",np.shape(train_images[3])[1], "images (For each emotion)"
print "Image size is", train_images[0][0].shape
print "Number of optimal projection axis is:", num_Eig









                                             

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

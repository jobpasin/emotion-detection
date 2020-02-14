#Training Process

#!/usr/bin/python
import os, sys
Original_Dir = os.getcwd()
#sys.path.append('/home/pi/.virtualenvs/cv/lib/python2.7/site-packages')
import timeit
import cv2
import numpy as np
from PIL import Image
sys.path.append(Original_Dir)

tic = timeit.default_timer()

#-------------------Initialize ----------------------------------
#PCA_Path = './TrainedData/1DPCA_Final_A16.npz'   #Select PCA path
PCA_Path = './TrainedData/new_test_6.npz'   #Select PCA path
path = './new test'                #Select tested picture folder
img_indexes = []                #Select specific image
emotions = ['hap','sad','ang','sur']                            #Select emotion to be detected
#----------------------------------------------------------------

def import_image(fullpath): #Importing single image
    image = np.matrix(np.array(Image.open(fullpath).convert('L'),'float32'))
    return image
    
def TestImage(image_temp, Eig, Im_Av): #Process of testing
    #stic = timeit.default_timer()
    # Number of eigenfaces, normalized image
    num_Eig_Test = Eig.shape[1]
    image_temp = np.reshape(image_temp,np.shape(image_temp)[0]*np.shape(image_temp)[0],'C').transpose()
    Test_Image = image_temp-Im_Av

    #Create projection face from training eigenface
    Test_weight = Eig.transpose()*Test_Image
    Projected_Image = Eig*Test_weight #Principle component of image using optimal projection

    #stoc = timeit.default_timer()
    #print "Time for create PCA from training eigenface is", stoc-stic, "seconds"
    
    #OpenImage(Eig,PCA_Train)
              
    #Calculate total euclidean distance
    error = np.linalg.norm(Test_Image - Projected_Image)
    #stoc2 = timeit.default_timer()
    #print "Time for finding errors", stoc2-stoc, "seconds"
    return error,Test_weight

def OpenImage(Eig,PCA):
    #Eig = np.matrix(Eig)
    PCA = np.matrix(PCA)
    image = PCA[:,0]*Eig[:,0].transpose()
    for cnt in range(1,Eig.shape[1]):
        image = image + PCA[:,cnt]*Eig[:,cnt].transpose()
    img = Image.fromarray(np.array(image),'L')
    img.save('myimage.png')
    img.show()
    
def ReconIm(Eig,num_Eig,Mode,*Path):  #Return eigenface into image
    #ReconIm(Eig[0],0,'E') or RecomIm(Im_Av[0],0,'Av')
    
    if Mode == "E":
        Eig_Im = np.reshape(Eig[:,num_Eig],(100,100))
        Eig_Im = Eig_Im-np.matrix.min(Eig_Im)
        Eig_Im = np.array(Eig_Im/np.max(Eig_Im)*255, dtype = np.uint8)
    elif Mode == "E2": #Inverted
        Eig_Im = np.reshape(Eig[:,num_Eig],(100,100))
        Eig_Im = -Eig_Im
        Eig_Im = Eig_Im-np.matrix.min(Eig_Im)
        Eig_Im = np.array(Eig_Im/np.max(Eig_Im)*255, dtype = np.uint8)        
    elif Mode == "Av":
        Eig_Im = np.reshape(Eig,(100,100))
        Eig_Im = np.array(Eig_Im, dtype = np.uint8)
    elif Mode == "S":
        Eig_Im = np.reshape(Eig[:,num_Eig],(100,100))
        Eig_Im = Eig_Im-np.matrix.min(Eig_Im)
        Eig_Im = np.array(Eig_Im/np.max(Eig_Im)*255, dtype = np.uint8)
        img = Image.fromarray(np.array(Eig_Im),'L')
        img.save(Path[0])
    else:
        print "Error mode : 'E' or 'Av'"
    Eig_Im = cv2.resize(Eig_Im,(400,400)) 
    cv2.imshow("Eigenface Image: "+str(num_Eig),Eig_Im)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def ProjIm(Im_Test,num_Eig,Eigs,Im_Avs): 
    Old_Image = cv2.resize(np.array(Im_Test,dtype = np.uint8),(400,400))
    cv2.imshow("Original Image:",Old_Image)
    cv2.moveWindow("Original Image:",0,0)
    emotions = [0,1,2,3]
    posX = [405, 810, 405, 810]
    posY = [0, 0, 405, 405]
    for emotion in emotions:
        # Number of eigenfaces, normalized image
        Eig = Eigs[emotion]
        Im_Av = Im_Avs[emotion]
        [H,W] =  Im_Test.shape
        image_temp = np.reshape(Im_Test,H*W,'C').transpose()
        Test_Image = image_temp-Im_Av
        Eig = Eig[:,0:num_Eig]
        #Create projection face from training eigenface
        Test_weight = Eig.transpose()*Test_Image
        Projected_Image = Eig*Test_weight #Principle component of image using optimal projection

        #Reconstruct images for showing image
        New_Image = Projected_Image+Im_Av
        New_Image = np.reshape(New_Image,(H,W),'C')
        New_Image = New_Image-np.min(New_Image)
        New_Image = New_Image/np.max(New_Image)*255
        New_Image = np.array(New_Image,dtype = np.uint8)
        New_Image = cv2.resize(New_Image,(400,400))
        cv2.imshow("Projected Image with #eigen"+str(num_Eig)+","+str(emotion),New_Image)
        cv2.moveWindow("Projected Image with #eigen"+str(num_Eig)+","+str(emotion),posX[emotion],posY[emotion])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return New_Image

def SaveIm(Im_Test,num_Eig,Eigs,Im_Avs,count):
    Old_Image = cv2.resize(np.array(Im_Test,dtype = np.uint8),(400,400))
    img = Image.fromarray(np.array(Old_Image),'L')
    img.save('./SavedImage/Original'+str(count)+'.jpg')
    emotions = [0,1,2,3]
    emotion_name = ['hap','sad','ang','sur']
    for emotion in emotions:
        # Number of eigenfaces, normalized image
        Eig = Eigs[emotion]
        Im_Av = Im_Avs[emotion]
        [H,W] =  Im_Test.shape
        image_temp = np.reshape(Im_Test,H*W,'C').transpose()
        Test_Image = image_temp-Im_Av
        Eig = Eig[:,0:num_Eig]
        #Create projection face from training eigenface
        Test_weight = Eig.transpose()*Test_Image
        Projected_Image = Eig*Test_weight #Principle component of image using optimal projection

        #Reconstruct images for showing image
        New_Image = Projected_Image+Im_Av
        New_Image = np.reshape(New_Image,(H,W),'C')
        New_Image = New_Image-np.min(New_Image)
        New_Image = New_Image/np.max(New_Image)*255
        New_Image = np.array(New_Image,dtype = np.uint8)
        New_Image = cv2.resize(New_Image,(400,400))
        img = Image.fromarray(np.array(New_Image),'L')
        img.save('./SavedImage/'+emotion_name[emotion]+str(count)+'.jpg')
    return    
#----------------Main Code -----------------------------#
tic = timeit.default_timer()
#Import data from training
PCAData = np.load(PCA_Path)
Eig = list()
Im_Av = list()
for cnt in range (0,4):
    Eig.append(np.matrix(PCAData['Eig'+str(cnt)]))
    Im_Av.append(np.matrix(PCAData['ImAv'+str(cnt)]))
toc1 = timeit.default_timer()
#print "Total time spend for importing data is:", toc1-tic, "seconds"
Im_Test = range(0,100)
num_Pic = 0
for emotion in emotions:
    if len(img_indexes) == 0:
        #image_paths = [os.path.join(path, f) for f in os.listdir(path) \
        #               if f.split(".")[0].endswith(emotion)]
        image_paths = [os.path.join(path, f) for f in os.listdir(path) \
                       if f.split("_")[0].endswith(emotion)]
    else:
        image_paths = list()
        for img_index in img_indexes:
            image_path = [os.path.join(path,f) for f in os.listdir(path) \
                            if (f.split("_")[0].endswith(emotion) and f.split("_")[1].endswith(img_index))]
            if image_path == []:
                print "Error finding data in path:" ,img_index, "from" ,img_indexes
            image_paths.append(image_path[0])   
    print "Emotion to be detected:", emotion
    image_paths.sort()
    for image_path in image_paths:
        #Import testing image
        Im_Test[num_Pic] = import_image(image_path)
        #happy/sad/normal/surprised = Happy/Sad/Angry/Surprised

        #Determine error for each class
        Error = list()
        if image_path.split('_')[1] == '1006.jpg':
            Test_Weight = list()
        MinError = 100000000
        MinEmotion = 0                                             
        for cnt in range(0,4):                                         
            Error_Temp,Weight_Temp = TestImage(Im_Test[num_Pic],Eig[cnt],Im_Av[cnt])
            Error.append(Error_Temp)
            if image_path.split('_')[1] == '1006.jpg':
                Test_Weight.append(Weight_Temp)
            #print "Error for emotion", cnt, "is", Error_Temp
            if Error_Temp < MinError:
                MinError = Error_Temp
                MinEmotion = cnt
        num_Pic = num_Pic + 1
        if MinEmotion == 0: 
            Emotion = 'Happy'
        elif MinEmotion == 1:
            Emotion = 'Sad'                                             
        elif MinEmotion == 2:
            Emotion = 'Angry'
        else:
            Emotion = 'Surprised'
        
        print "Emotion detected for image",os.path.split(image_path)[1] ,"is:", Emotion, " with error", Error
        
    
toc = timeit.default_timer()
print "Total time spend for testing is:", toc-tic, "seconds"
print "Total image tested :", len(image_paths), "images"
print "Emotion detected is", Emotion














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

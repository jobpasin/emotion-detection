
#Training Process

import os, sys
Original_Dir = os.getcwd()
#sys.path.append('/home/pi/.virtualenvs/cv/lib/python2.7/site-packages') #######
import random
import timeit
import cv2
import numpy as np
from PIL import Image
sys.path.append(Original_Dir)

#-------------------Initialize ----------------------------------
Path = 1
Use_testpic = 0                     #0 = Use remaining train picture as test picture
                                    #1 = Use path_Test as test picture



if Path == 1:
    path_Train = './TrainingPicsAll'   #Training picture path
    path_Test = './TestingPics_All' #Testing picture path
    Data_path = './Report/Data1D_Valid_TestAA32.txt'        #Text data saving path ##########
elif Path == 2:
    path_Train = './NitadDB/TrainingA'   #Training picture path
    path_Test = './NitadDB/TestingA'     #Testing picture path
    Data_path = './Report/1D_Nitad_Valid_A_A.txt'        #Text data saving path ##########
else:
    print('Path Error')
TrainWithGlass = 0                  #0 = No glasses, #1 = Glasses only, #2= All
num_TrainPic = 14                   #Max number of training pic (start at 3)
num_Eig_Max = 14                    #Max number of Eig (start at 1)
num_start_max = 0                   #Ignore some highest eigenvectors
num_Repeat = 300                    #Repeat the process
Happy = 'hap'                       #Name of each emotion
Sad = 'sad'
Angry = 'ang'
Surprised = 'sur'
#Happy = 'happy'
#Sad = 'sad'
#Angry = 'wink'
#Surprised = 'surprised'

#----------------------------------------------------------------


def import_images(path,num_Pic,Use_testpic,path_Test,TrainWithGlass): #Process of importing images from a folder
    image_vec = range(4)
    image_test_vec = range(4)
    im_av = range(4)
    image_paths = range(4)
    emotion = ['hap','sad','ang','sur']
    for i in range(0,4):
        #Insert image path with the interested emotion
        #image_paths = [os.path.join(path, f) for f in os.listdir(path) \
        #               if f.split(".")[0].endswith(emotion)] #-------
        if TrainWithGlass == 0:
            image_paths[i] = [os.path.join(path, f) for f in os.listdir(path) \
                           if f.split("_")[0].endswith(emotion[i]) and f[-7:].startswith("0")]
        elif TrainWithGlass == 1:
            image_paths[i] = [os.path.join(path, f) for f in os.listdir(path) \
                           if f.split("_")[0].endswith(emotion[i]) and f[-7:].startswith("1")]
        else:
            image_paths[i] = [os.path.join(path, f) for f in os.listdir(path) \
                           if f.split("_")[0].endswith(emotion[i])]
        image_paths[i].sort()
        #print(image_paths[i])
        # Find size of an image 
        image_pil = Image.open(image_paths[0][0]).convert('L')
        [H,W] = np.shape(np.array(image_pil,'float32'))   
        #num_train_pic = len(image_paths)
        im_av[i] = np.array(np.zeros([H*W,1],'float32')) #Create average value
    #Random selected num_Pic images
    random.shuffle(image_paths[0])
    img_index = image_paths[0][0:num_Pic]
    img_index.sort()
    #print(img_index)

    index_name = [i.split('_',1)[1] for i in img_index]

    for i in range(0,4):
        #print(image_paths[i])
        for index in img_index:
            #Read the selected images from folder ,convert to grayscale matrix
            image_path_temp = [f for f in image_paths[i] if f.split("_")[1].endswith(index.split("_")[1])]
            if image_path_temp == []:
                print("Error finding data in path:" ,index, "from" ,image_paths[i])
            image_path = image_path_temp[0]
            #print("Train:",image_path)
            image_paths[i].remove(image_path)
            image_pil = Image.open(image_path).convert('L')
            image = np.matrix(np.array(image_pil,'float32'))
            image_vec_temp = np.reshape(image,H*W,'C').transpose()
            if index == img_index[0]:
                image_vec[i] = image_vec_temp
            else:
                image_vec[i] = np.concatenate((image_vec[i],image_vec_temp),axis=1)
            im_av[i] = im_av[i]+image_vec_temp
        im_av[i] = im_av[i]/num_Pic
        image_vec[i] = image_vec[i] - np.tile(im_av[i],(1,num_Pic))
        # Testing pictures -------------------------------------------------------------
        pic_cnt = 0
        image_test_vec[i] = list()
        if Use_testpic == 1:
            image_paths[i] = []
            image_paths[i] = [os.path.join(path_Test, f) for f in os.listdir(path_Test) \
                           if f.split("_")[0].endswith(emotion[i])]
        for image_path in image_paths[i]:
            # Read the image and convert to grayscale, convert to matrix
            #print("Test:", image_path)
            image_test_pil = Image.open(image_path).convert('L')
            image_test = np.matrix(np.array(image_test_pil,'float32'))
            image_test_vec_temp = np.reshape(image_test,H*W,'C').transpose()
            image_test_vec[i].append(image_test_vec_temp)
            pic_cnt = pic_cnt + 1
            if (pic_cnt == len(image_paths[i])):
                break
        
        # return the images list and average
        
    return image_vec, image_test_vec ,im_av,index_name

def create_Eig(images,im_av,num_Eig,num_Ignore): #Process of creating optimal projection axis
    
    [Size,num_pic] = images.shape # Number of training images
    if num_pic < num_Eig:
        num_Eig = num_pic
        print("!!Eigenfaces is limited to number of training images")
    #Create pseudo covariance matrix
    Cov = images.transpose()*images

    #Compute Eigenvalue and Eigenvector
    [EgVal,EgVec_temp] = np.linalg.eig(Cov)
    EgVec_temp = np.matrix(EgVec_temp)
    EgVec = images*EgVec_temp
    #Rearrange from highest eigenvalue to lowest
    index = EgVal.argsort()[::-1]
    EgVal = EgVal[index]
    EgVec = EgVec[:,index]
    #Select only num_Eig vectors and normalize to size of 1
    Eig = EgVec[:,num_Ignore]/np.linalg.norm(EgVec[:,num_Ignore])
    for cnt in range(num_Ignore+1,num_Eig+num_Ignore):
        Eig = np.concatenate((Eig,EgVec[:,cnt]/np.linalg.norm(EgVec[:,cnt])),axis=1)
    # return optimal projection axis and average image for future use
    Sum_EgVal = np.sum(EgVal,dtype = 'float32')
    EgVal_percent = EgVal/Sum_EgVal*100
    return Eig,EgVal_percent[num_Ignore:num_Ignore+num_Eig]

def import_image(fullpath): #Importing single image
    image = np.matrix(np.array(Image.open(fullpath).convert('L'),'float32'))
    [H,W] = np.shape(image)
    image_vec = np.reshape(image,H*W,'C').transpose()
    return image_vec
    
def TestImage(image_temp, Eig, Im_Av): #Process of testing
    #stic = timeit.default_timer()
    # Number of eigenfaces, normalized image
    num_Eig_Test = Eig.shape[1]
    image_temp = np.reshape(image_temp,np.shape(image_temp)[0]*np.shape(image_temp)[1],'C').transpose()
    Test_Image = image_temp-Im_Av

    #Create projection face from training eigenface
    Test_weight = Eig.transpose()*Test_Image
    Projected_Image = Eig*Test_weight #Principle component of image using optimal projection

    #stoc = timeit.default_timer()
    #print ("Time for create PCA from training eigenface is", stoc-stic, "seconds")
    
    #OpenImage(Eig,PCA_Train)
              
    #Calculate total euclidean distance
    error = np.linalg.norm(Test_Image - Projected_Image)
    #stoc2 = timeit.default_timer()
    #print ("Time for finding errors", stoc2-stoc, "seconds")
    return error


#----------------Setup Code -----------------------------#


File = open(Data_path,'w+') 
File2 = open("."+Data_path.split(".")[1]+"_summary."+Data_path.split(".")[2],'w+') ###############

File.write("Error%Happy/Correction%Sad/#PCA%Angry/#Ignore%Surprised/#Train%Real Emotion/Loop%Detect as/EigenValue%# PCA%Ignore%Num of Training picture\n") 
#print ("Number of optimal projection axis is:", num_optproj)
#----------------Train Code -----------------------------#
for num_Pic in range(13,num_TrainPic+1):
    print ("Num Pic:",num_Pic)
    for num_loop in range(0,num_Repeat):
        # Importing image
    ##    train_images = range(4)
    ##    test_images =range(4)
    ##    Im_Av = range(4)
        train_images, test_images,Im_Av,Im_Index = import_images(path_Train,num_Pic,Use_testpic,path_Test,TrainWithGlass)
        File.write("Random samples : "+str(Im_Index)+"Num_Pic is%"+str(num_Pic)+"%Loop:%"+str(num_loop)+"\n")
        for num_Ignore in range(0,num_start_max+1):
            for num_Eig in range (1,min(num_Eig_Max+1,num_Pic-num_Ignore)):
            #for num_Eig in range (1,2):
                # Create Eigenface 
                Eig = range(4)
                EigVal = range(4)
                for cnt in range(0,4):
                    Eig[cnt],EigVal[cnt] = create_Eig(train_images[cnt],Im_Av[cnt],num_Eig,num_Ignore)                                   
                #print ("Total images for training is", len(train_images[0]), ",",len(train_images[1]),",",\
                #      len(train_images[2]),",",len(train_images[3]), "images (For each emotion)")

                #----------------Test Code -----------------------------#
                #print ("Number of training picture:", num_Pic, "pictures")
                #print ("Number of PCA:", num_Eig)
                #print ("Ignore", num_Ignore, "eigenvectors")
                File2.write("Num of training picture: %"+str(num_Pic)+"%# of PCA: %"+str(num_Eig)+"% Ignore: %"+str(num_Ignore)+"\n")
                correct_pic = 0
                total_pic = 0
                for emotion_type in range(0,4):
                    correct_pic_temp = 0
                    total_pic_temp = 0
                    if emotion_type == 0:
                        emotion = Happy
                    elif emotion_type == 1:
                        emotion = Sad
                    elif emotion_type == 2:
                        emotion = Angry
                    else:
                        emotion = Surprised
                    #print ("Number of testing image ", emotion, "is:", len(test_images[emotion_type]), "pictures")
                    for test_image in test_images[emotion_type]:
                        #happy/sad/normal/surprised = Happy/Sad/Angry/Surprised

                        #Determine error for each class
                        Error = list()
                        MinError = 100000000
                        MinEmotion = 0                                             
                        for cnt in range(0,4):                                         
                            Error_Temp = TestImage(test_image,Eig[cnt],Im_Av[cnt])
                            Error.append(Error_Temp)
                            #print ("Error for emotion", cnt, "is", Error_Temp)
                            if Error_Temp < MinError:
                                MinError = Error_Temp
                                MinEmotion = cnt
                        if MinEmotion == emotion_type:
                            correct_pic_temp = correct_pic_temp + 1
                        total_pic_temp = total_pic_temp + 1
                        #print ("Error for this image is:", Error,"\n", "Detect as emotion number: ", MinEmotion)
                        #print ("Detect as emotion number: ", MinEmotion, "Real emotion is: ", emotion_type , 'Error :', MinError)
                        File.write("Error :%"+str(Error[0])+"%"+str(Error[1])+"%"+str(Error[2])+"%"+str(Error[3])
                                   +"%"+str(emotion_type)+"%"+str(MinEmotion)+"%"+str(num_Eig)+"%"+str(num_Ignore)+"%"+str(num_Pic)+"\n") 
                    correct_pic = correct_pic+correct_pic_temp
                    total_pic = total_pic+total_pic_temp
                    #print ("%Correction for emotion: ", emotion, str(correct_pic_temp/float(total_pic_temp)*100), "%")
                    File.write("Correction for "+emotion+" is%" + str(correct_pic_temp/float(total_pic_temp)*100)+"%"+str(num_Eig)+"%"
                               +str(num_Ignore)+"%"+str(num_Pic)+"%"+str(num_loop)+"%"+np.array_str(EigVal[emotion_type],100000)+"%"+str(np.sum(EigVal[emotion_type]))+" %\n")
                    File2.write("Correction for "+emotion+" is%" + str(correct_pic_temp/float(total_pic_temp)*100)+" %\n")
                #print ("Total correction is: ",str(correct_pic/float(total_pic)*100), "%\n")
                File.write("Total accuracy is%" + str(correct_pic/float(total_pic)*100)+"%"+str(num_Eig)+"%"+str(num_Ignore)+"%"+str(num_Pic)+"%"+str(num_loop)+"%\n")
                File2.write("Total correction is%" + str(correct_pic/float(total_pic)*100)+" %\n")  
                        ##if MinEmotion == 0: 
                        ##    Emotion = 'Happy'
                        ##elif MinEmotion == 1:
                        ##    Emotion = 'Sad'                                             
                        ##elif MinEmotion == 2:
                        ##    Emotion = 'Angry'
                        ##else:
                        ##    Emotion = 'Surprised'
                        ##print ("Emotion detected for image",os.path.split(image_path)[1] ,"is:", Emotion, " with error", MinError)
                    
                #toc2 = timeit.default_timer()

                #print ("Total time spend for testing is:", toc2-tic, "seconds")
                #print ("Total image tested :", len(image_paths), "images")
                #print ("Emotion detected is", Emotion)
File.write("**Ended Process**")
File.close()
File2.write("**Ended Process**")
File2.close()

print ("1DPCA Validation Finished")
if TrainWithGlass == 0:
    Message = 'Train with No Glasses'
elif TrainWithGlass == 1:
    Message = 'Train with Glasses'
else:
    Message = 'Train with all images'
if Use_testpic == 1:
    print (Message,"Test using", path_Test)
else:
    print (Message,"Test using remaining images")
print ("Save at ", Data_path)
                                             

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

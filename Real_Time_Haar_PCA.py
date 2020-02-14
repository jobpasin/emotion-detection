#Real-time detection with all outputs
#Import necessary library
import os, sys
Original_Dir = os.getcwd()
sys.path.append('/home/pi/.virtualenvs/cv/lib/python2.7/site-packages')
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera, Color
from time import sleep
from gpiozero import LED
from PIL import Image
import cv2
import time
sys.path.append(Original_Dir)


#----------------------- Initial setup ----------------------------------
PCAData = np.load('./TrainedData/new_test_7.npz')       #Choose trained data
AutoMode = 0                                            #1 = Real-time detection
PicCnt_path = './NitadPics/PicCnt.txt'                  
PicSave_path = './NitadPics/'
#-------------------------------------------------------------------------

#Initialize hardware parameters
#LED
led = LED(17)
red = LED(10)
blue = LED(9)
green = LED(11)
led.on()
red.on()
blue.on()
green.on()
LED_Cnt = 0             #Counting times before LED is off
Button_Cnt = 0          #Delay time before button can be pressed again
ButtonEnabled = 1       #Button can be pressed

#PiCamera
camera = PiCamera()
camera.resolution=(528,384)
camera.rotation = 180
camera.framerate= 20
camera.annotate_text_size = 30
rawCapture = PiRGBArray(camera,size=(528,384))

#LCD screen output face 
Im_Output = [cv2.imread('/home/pi/NITAD17/Face_expression/Happy.jpg'),
             cv2.imread('/home/pi/NITAD17/Face_expression/Sad.jpg'),
             cv2.imread('/home/pi/NITAD17/Face_expression/Angry.jpg'),
             cv2.imread('/home/pi/NITAD17/Face_expression/Surprised.jpg')]

#Testing function 
def import_image(fullpath): #Importing single image
    image = np.array(Image.open(fullpath).convert('L'),'float32')
    return image

def TestImage(image_temp, Eig, Im_Av): #Process of testing
    num_Eig_Test = Eig.shape[1]
    image_temp = np.matrix(np.reshape(image_temp,np.shape(image_temp)[0]*np.shape(image_temp)[0],'C')).transpose()
    Test_Image = image_temp-Im_Av

    #Create projection face from training eigenface
    Test_weight = Eig.transpose()*Test_Image
    Projected_Image = Eig*Test_weight 

    #Calculate total euclidean distance
    error = np.linalg.norm(Test_Image - Projected_Image)
    return error

#--------------Main Code -----------------------------#
print 'Running... please hold "s" to stop program'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Import Haar-like trained data
time.sleep(1)
if AutoMode == 1: 
    print 'Auto detection on'
else:
    print 'Press "r" to capture face'    

#Import Eigenfaces and average images
Eig = list()
Im_Av = list()
for cnt in range (0,4):
    Eig.append(np.matrix(PCAData['Eig'+str(cnt)]))
    Im_Av.append(np.matrix(PCAData['ImAv'+str(cnt)]))

#Start capturing video real-time 
for frame in camera.capture_continuous(rawCapture,format="bgr",use_video_port=True):
    img = frame.array
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray[:,0:425], 1.3, 5) #Haar-like detection
    for (x,y,w,h) in faces :
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray =  gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
    cv2.imshow('Detecting face...',cv2.resize((img),(960,720))) #Display detected image
    cv2.moveWindow('Detecting face...',0,0)
    rawCapture.truncate(0)
    if faces != ():
        if AutoMode == 1:   #If automode is on, process is real time
            #Cropped & Histrogram equalization
            cropped_img = img[y:y+h , x:x+w]
            grayscale_img = cv2.resize(cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY),(100,100))
            equ = cv2.equalizeHist(grayscale_img)
            hist_eq_img = np.hstack((grayscale_img,equ))
            new_img = hist_eq_img[0:100, 100:200]
            #Determine error for each class using PCA
            Error = list()
            MinError = 100000000
            MinEmotion = 0                                             
            for cnt in range(0,4):                                         
                Error_Temp = TestImage(new_img,Eig[cnt],Im_Av[cnt])
                Error.append(Error_Temp)
                if Error_Temp < MinError:
                    MinError = Error_Temp
                    MinEmotion = cnt
            #Display output based on emotion with minimum error
            if MinEmotion == 0: 
                Emotion = 'Happy' #Dark Blue
                red.on()
                blue.off()
                green.on()
            elif MinEmotion == 1: #Red
                Emotion = 'Sad'
                red.off()
                blue.on()
                green.on()           
            elif MinEmotion == 2: #Yellow
                Emotion = 'Angry'
                red.off()
                blue.on()
                green.off()
            else:
                Emotion = 'Surprised' #Sky blue
                red.on()
                blue.off()
                green.off()
            LED_Cnt = 0
            cv2.imshow('Face Cropped',cv2.resize((new_img),(180,180)))
            cv2.moveWindow('Face Cropped',970,360)
            cv2.imshow('Emotion',cv2.resize(Im_Output[MinEmotion],(250,300)))
            cv2.moveWindow('Emotion',970,37)0) 
            cv2.waitKey(1)
            print "Emotion detected:", Emotion 
        elif cv2.waitKey(1) & 0xFF == ord('r'): #If automode is off, button 'r' must be pressed to capture
            #Cropped & Histrogram equalization
            cropped_img = img[y:y+h , x:x+w]
            grayscale_img = cv2.resize(cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY),(100,100))
            equ = cv2.equalizeHist(grayscale_img)
            hist_eq_img = np.hstack((grayscale_img,equ))
            new_img = hist_eq_img[0:100, 100:200]
            #Determine error for each class using PCA
            Error = list()
            MinError = 100000000
            MinEmotion = 0                                             
            for cnt in range(0,4):                                         
                Error_Temp = TestImage(new_img,Eig[cnt],Im_Av[cnt])
                Error.append(Error_Temp)
                if Error_Temp < MinError:
                    MinError = Error_Temp
                    MinEmotion = cnt
            #Display output based on emotion with minimum error
            if MinEmotion == 0: 
                Emotion = 'Happy' #Dark Blue
                red.on()
                blue.off()
                green.on()
            elif MinEmotion == 1: #Red
                Emotion = 'Sad'
                red.off()
                blue.on()
                green.on()           
            elif MinEmotion == 2: #Yellow
                Emotion = 'Angry'
                red.off()
                blue.on()
                green.off()
            else:
                Emotion = 'Surprised' #Sky blue
                red.on()
                blue.off()
                green.off()
            LED_Cnt = 0
            cv2.imshow('Face Cropped',cv2.resize((new_img),(180,180)))
            cv2.moveWindow('Face Cropped',970,360)
            cv2.imshow('Emotion',cv2.resize(Im_Output[MinEmotion],(250,300)))
            cv2.moveWindow('Emotion',970,37)) 
            cv2.waitKey(1)
            print "Emotion detected:", Emotion
    else:
        if LED_Cnt == 20:
            #Turn off light when there is no face detected for too long
            red.on()
            blue.on()
            green.on()
            LED_Cnt = 0
        LED_Cnt = LED_Cnt+1
    Button = cv2.waitKey(1) & 0xFF
    if Button == ord('s'): #Press 's' to stop process
        break
    elif Button == ord('o') and ButtonEnabled == 1: #Press 'o' to stop automode
        AutoMode = 0
        ButtonEnabled = 0
        print 'AutoMode off'
        print 'Press "r" to capture face'
        print 'Then, press "1","2","3","4","5" to save images as Happy,Sad,Angry,Surprised,Neutral image'
    elif Button == ord('p') and ButtonEnabled == 1: #Press 'p' to start automode
        AutoMode = 1
        ButtonEnabled = 0
        print 'AutoMode on'
    if ButtonEnabled == 0:
        if Button_Cnt == 10:
            ButtonEnabled = 1
            Button_Cnt = 0
        else:
            Button_Cnt = Button_Cnt + 1
print 'End process. Please run again to restart program'
camera.close()
cv2.destroyAllWindows()

import cv2
import numpy as np
import pickle
import sys
import os


face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')



def face_detector(frame):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None,None

    roi = []
    cord = []    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.rectangle(frame, (x,y-30), (x+w,y), (0,255,255), -1)
        cv2.putText(frame, 'None', (x+2,y-10), 4, 0.6, (255,255,255), 1)
        # cv2.circle(frame, (x+130,y), 1, (0,0,255), -1)
        croi = frame[y:y+h, x:x+w]
        l = []
        l.append((x,y))
        l.append((x+w,y+h))
        cord.append(l)
        roi.append(cv2.resize(croi, (200, 200)))
    return roi,cord





if __name__ == '__main__':

    totalModel = 0

    with open('train/FaceData/imagecount.txt','r') as f:
        totalModel = int(f.read())+1

    modelNameList = []

    for i in range(totalModel):
        modelNameList.append('train/FaceData/'+str(i)+'.cv2')

    nameList = []

    modelList = []

    pmodelList = []

    for i in range(totalModel):
        modelList.append(cv2.createLBPHFaceRecognizer())
        # print(modelList[i])

    for i in range(totalModel):
        modelList[i].load(modelNameList[i])    

    with open('train/FaceData/nameList','r') as f:
                nameList = pickle.load(f)        

    # print(nameList)

    try:             

        if sys.argv[1]=='-dir':
            if os.path.exists(sys.argv[2]):
                frame = cv2.imread(sys.argv[2])

                frame = cv2.resize(frame,(500,400))

                
                faces,positions = face_detector(frame)

                if faces is None:
                    cv2.imshow('Face Recognition', frame )
                else:    
                    try:
                        for i,face in enumerate(faces):
                            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                            for j,model in enumerate(modelList):
                                results = model.predict(face)
                            
                                if results[1] < 500:
                                    confidence = int( 100 * (1 - (results[1])/400) )
                                    if confidence>=70:
                                        cv2.rectangle(frame, (positions[i][0][0],positions[i][0][1]-30), (positions[i][1][0],positions[i][0][1]), (0,0,255), -1)
                                        cv2.putText(frame, nameList[j], (positions[i][0][0]+2,positions[i][0][1]-10), 4, 0.6, (255,255,255), 1)
                                        cv2.rectangle(frame, positions[i][0], positions[i][1], (0,0,255), 2)
                                        break

                    except:
                        raise

                cv2.imshow('Face Recognition', frame )
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                raise    


    
        elif sys.argv[1]=='-cam':

            camNumber = 0

            try:
                camNumber =sys.argv[2]
            except:
                pass    

            cap = cv2.VideoCapture(camNumber)

            while True:

                ret, frame = cap.read()

                frame = cv2.resize(frame, (500,400))
                
                faces,positions = face_detector(frame)

                if faces is None:
                    cv2.imshow('Face Recognition', frame )
                else:    
                    try:
                        for i,face in enumerate(faces):
                            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                            for j,model in enumerate(modelList):
                                results = model.predict(face)
                            
                                if results[1] < 500:
                                    confidence = int( 100 * (1 - (results[1])/400) )
                                    if confidence>=75:
                                        cv2.rectangle(frame, (positions[i][0][0],positions[i][0][1]-30), (positions[i][1][0],positions[i][0][1]), (0,0,255), -1)
                                        cv2.putText(frame, nameList[j], (positions[i][0][0]+2,positions[i][0][1]-10), 4, 0.6, (255,255,255), 1)
                                        cv2.rectangle(frame, positions[i][0], positions[i][1], (0,0,255), 2)
                                        break

                    except:
                        raise

                cv2.imshow('Face Recognition', frame )            
                    
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
                    
            cap.release()
            cv2.destroyAllWindows()     

        else:
            raise    

    except:
        print('Please pass arguments , while running program.')     
        print("-dir Absolute_Path_To_Testing_Image_File_Is")
        print("-cam Camera_Number_Default_Is_Set_To_0")
        print('----------')
        raise
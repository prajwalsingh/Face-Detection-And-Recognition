# Face Recognition With Name

* Identify different Faces With Name with the help of training data.

* Algorithm used is LBPHFaceRecognizer which is available in opencv 2.4.11.

* [Read About LBPHFaceRecognizer](http://eyalarubas.com/face-detection-and-recognition.html)

* This is predictor based classifier , so sometimes result may varies based on how clear training data images are.

#### Require Library

* `cv2`
* `numpy`
* `pickle`

#### How To Use :
* `Train face classifier with help of` **CaptureAndTrain.py** `python file.`
* **$python CaptureAndTrain.py -pn 'NameOfPerson' -dir Absolute_Path_To_Directory_Where_Training_Image_Is_Present** 
* `Use above command when you already have more than 50 sample images for training.`
* **$python CaptureAndTrain.py -pn 'NameOfPerson' -cam Camera_Number_Default_Is_Set_To_0**
*  `Use above command when you want to train data with help of PC Camera.It takes 100 sample photos, of person present in front of camera.`
* `In last it will show Successfull message.`
* `After running above file now run` **CheckFace.py** `file to recognize face with name.`
* **$python CheckFace.py -dir Absolute_Path_To_Testing_Image_File_Is**
* `Use above command when you want to recognize image from saved photo.`
* **$python CheckFace.py -cam Camera_Number_Default_Is_Set_To_0**
* `Use above command when you want live recognization with help of PC camera.`

<br/>
<br/>

**Sample Output**

![alt](https://github.com/prajwalsingh/SimpleFaceClassifier/blob/master/SampleImage.png)

![alt](https://github.com/prajwalsingh/SimpleFaceClassifier/blob/master/SimpleFaceClassifier.png)

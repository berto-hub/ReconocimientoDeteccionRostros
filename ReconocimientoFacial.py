import glob
import cv2
import numpy as np

images = [cv2.imread(image) for image in glob.glob("imagesTest/keanu_prueba2.jpg")]

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Se lee el modelo
face_recognizer.read('modeloLBPHFace.xml')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for image in images:
    #Convertir a escala de grises
    img_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Hacer equalize the histogram
    img_eqHist = cv2.equalizeHist(img_gris)
    auxFrame = img_eqHist.copy()
    faces_detected = face_cascade.detectMultiScale(img_eqHist, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)) 

    #visualización rectángulo
    for i in faces_detected:
        (x, y, w, h) = i
        user, result = face_recognizer.predict(img_eqHist[y:y+h,x:x+w])

        cv2.putText(img_eqHist,'{}'.format(result),(x,y-5),1,1.1,(255,255,0),1,cv2.LINE_AA)
        print(result)
        if(result < 55):
            if(user == 0):
                cv2.putText(img_eqHist,'Keanu',(x,y-25),1,1.1,(255,255,0),1,cv2.LINE_AA)
            else:
                cv2.putText(img_eqHist,'Leonardo',(x,y-25),1,1.1,(255,255,0),1,cv2.LINE_AA)
            cv2.rectangle(img_eqHist, (x, y), (x + w, y + h), (255, 0, 0), 2); 
        else:
            cv2.putText(img_eqHist,'Desconocido',(x,y-25),1,1.1,(255,255,0),1,cv2.LINE_AA)
            cv2.rectangle(img_eqHist, (x, y), (x + w, y + h), (255, 0, 0), 2);
        
    cv2.imshow('Imagen con rostro reconocido',img_eqHist)

cv2.waitKey()
cv2.destroyAllWindows()
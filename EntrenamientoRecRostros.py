import cv2
import glob
import os
import numpy as np

num = 1
images = [cv2.imread(image) for image in glob.glob("images/*.jpg")]
grayImages = []
preproImages=[]
labels = []
label = 0
nombre = "_keanu"
facesData = []

for image in images:
    fileResultName = "resultImages"
    #Ejercicio 1
    #Convertir a escala de grises
    img_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Hacer equalize the histogram
    img_eqHist = cv2.equalizeHist(img_gris) 
    grayImages.append(img_gris)
    preproImages.append(img_eqHist)

    #cv2.imshow('Imagen original', image)
    #cv2.imshow('Gris', img_gris)
    #cv2.imshow('Gris Ecualización de Histograma', img_eqHist)

    #Ejercicio 2
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_detected = face_cascade.detectMultiScale(img_eqHist, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)) 

    if(num<=10):
        labels.append(label)
        fileResultName = fileResultName + nombre  + str(num)
    else:
        label = label + 1
        num = 1
        labels.append(label)
        nombre = "_leo"
        fileResultName = fileResultName + nombre + str(num)
    
    #visualización rectángulo
    for i in faces_detected:
        (x, y, w, h) = i
        cv2.rectangle(img_eqHist, (x, y), (x + w, y + h), (255, 0, 0), 2); 

    #cv2.imshow('Imagen con rostro detectado',img_eqHist)
    
    #Ejercicio3
    face_crop = img_eqHist[faces_detected[0][1]:faces_detected[0][1] + faces_detected[0][3],
                           faces_detected[0][0]:faces_detected[0][0] + faces_detected[0][2]]

    cv2.imwrite('resultImages/' + fileResultName + '.jpg', face_crop)
    num += 1
    facesData.append(face_crop)
    
    #cv2.imshow('Rostro detectado',face_crop)
    cv2.waitKey()

#Entrenamiento del modelo de reconocimiento de cara usando face_crop y labels
recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Entrenando...")
recognizer.train(facesData, np.array(labels))

recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado")
cv2.destroyAllWindows()

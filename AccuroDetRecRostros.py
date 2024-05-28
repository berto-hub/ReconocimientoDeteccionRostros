import cv2
import glob

#imagen = cv2.imread('grupo.jpg')
num = 1
images = [cv2.imread(image) for image in glob.glob("images/*.jpg")]

for image in images:
#Ejercicio 1
    img_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # equalize the histogram of the Y channel
    img_eqHist = cv2.equalizeHist(img_gris)
    #cv2.imwrite('resultEqHist.jpg',img_eqHist)

    #cv2.imshow('Imagen original', image)
    #cv2.imshow('Gris', img_gris)
    #cv2.imshow('Gris Ecualización de Histograma', img_eqHist)

#Ejercicio 2
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_detected = face_cascade.detectMultiScale(img_eqHist, scaleFactor=1.1, minNeighbors=4)
    
    fileResultName = "resultImages" + str(num)
    #visualización rectángulo
    for i in faces_detected:
        (x, y, w, h) = i
        cv2.rectangle(img_eqHist, (x, y), (x + w, y + h), (255, 200, 0), 1); 

    cv2.imwrite('resultImages/' + fileResultName + '.jpg',img_eqHist)
    num = num + 1

    cv2.imshow('Imagen con rostro detectado',img_eqHist)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

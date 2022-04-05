import string

import PIL.Image
import PIL.ImageDraw
import face_recognition
from matplotlib.image import imread
import cv2
import numpy as np
import pandas as pd


# load the image and convert it into numpy array
image = face_recognition.load_image_file("images/sample_image.jpg")


# find all the faces in the image
face_locations = face_recognition.face_locations(image)
no_of_faces = len(face_locations)
#print(no_of_faces)

# draw the rectangular image on the face

pil_image = PIL.Image.fromarray(image)
i=0
list=[]
for face_location in face_locations:
    top, right, bottom, left = face_location
    draw_shape = PIL.ImageDraw.Draw(pil_image)
    draw_shape.rectangle([left, top, right, bottom], outline="red")
    #*************************************************
    # crop img
    pil_image.save("images/output_image.jpg")
    img = cv2.imread("images/output_image.jpg")
    roi = img[top: bottom, left: right]
    cv2.imwrite("cut"+i.__str__()+".jpg",roi)
    list.append("cut"+i.__str__()+".jpg")
    i+=1
    #*************************************************
    # save images on excel file
    data = pd.DataFrame({'img' : [roi]})

    #toExcel = pd.ExcelWriter('cutImg.xlsx')
    #data.to_excel(toExcel,sheet_name='Sheet1')
    #toExcel.save()
    # *************************************************


# displat the image
#pil_image.show()
print(list)
cv2.waitKey(0)
#************************************************************
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\HP\\Downloads\\shape_predictor_68_face_landmarks.dat")
u=0
points=[]
data =[]
for i in list:

    image = face_recognition.load_image_file(i)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            u+=1
            #print(x , y)
            #print(u)
            points.append([x,y])

            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
    data.append([points,i])


    key = cv2.waitKey(1)
    if key == 27:
        break
print(data[0])
print("+===================================================================")
print(data[2])


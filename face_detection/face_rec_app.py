import numpy as np
import cv2
import face_recognition

img_elon=face_recognition.load_image_file('imageBasic/elonbasic.jpg')
img_elon=cv2.resize(img_elon,(600,400))
img_elon=cv2.cvtColor(img_elon,cv2.COLOR_BGR2RGB)

img_test=face_recognition.load_image_file('imageBasic/elontest.jpg')
img_test=cv2.resize(img_test,(600,400))
img_test=cv2.cvtColor(img_test,cv2.COLOR_BGR2RGB)



faceloc=face_recognition.face_locations(img_elon)[0]
encode_img=face_recognition.face_encodings(img_elon)[0]
cv2.rectangle(img_elon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest=face_recognition.face_locations(img_test)[0]
encode_imgTest=face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

result=face_recognition.compare_faces([encode_img],encode_imgTest)

faceDis=face_recognition.face_distance([encode_img],encode_imgTest)

print(result,faceDis)
cv2.imshow('basicimage',img_elon)
cv2.imshow('testimage',img_test)

cv2.waitKey(0)
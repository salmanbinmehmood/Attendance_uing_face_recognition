
import numpy as np
import cv2
import face_recognition
import os
from csv import writer
from datetime import datetime
import time
import datetime
images=[]

path='imageAttence'
classNames=[]
myList=os.listdir(path)

def markAttendance(name):
    global count

    if count%2==0:
        list_data = [name, 'present', datetime.datetime.now().strftime("%d-%b-%Y")]
        with open('attendance.csv', 'a', newline='') as f_object:
            # Pass the CSV  file object to the writer() function
            writer_object = writer(f_object)
            # Result - a writer object
            # Pass the data in the list as an argument into the writerow() function
            writer_object.writerow(list_data)
            # Close the file object
            f_object.close()


print(myList)
for cl in myList:
    curimg=cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList
findKwonList=findEncodings(images)
# print(len(findKwonList))
print('completed')

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

ch = True
count=0
while True:

    success,img=cap.read()
    resized_img=cv2.resize(img,(0,0),None,0.25,0.25)
    resized_img=cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    curFaceLoc = face_recognition.face_locations(resized_img)
    encodeCurFrame = face_recognition.face_encodings(resized_img,curFaceLoc)
    for encodeface,faceloc in zip(encodeCurFrame,curFaceLoc):

        matches = face_recognition.compare_faces(findKwonList,encodeface )

        faceDis = face_recognition.face_distance(findKwonList,encodeface )
        # print(matches)
        print(faceDis)
        print(matches)
        matchIndexes=np.argmin(faceDis)

        if matches[matchIndexes] :

            names=classNames[matchIndexes].upper()
            print(names)

            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1 ), (x2, y2), (255, 0, 255), 2)
            cv2.rectangle(img, (x1,y2 -35), (x2, y2), (255, 0, 255), cv2.FILLED)

            cv2.putText(img, names, (x1+6,y2-6), cv2.FONT_ITALIC, 1,
                        (255, 255, 255), 2)
            cv2.rectangle(img, (x1 + 100, y2 - 35), (x2 + 300, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, 'Attendance Marked', (x1 + 140, y2 - 6), cv2.FONT_ITALIC, 1,
                        (255, 255, 255), 2)
            time.sleep(1)
            count+=1
            markAttendance(names)

    cv2.imshow('img',img)

    k = cv2.waitKey(1) & 0XFF
    if k == 27:
        break


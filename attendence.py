from base64 import encode
import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
import os
from datetime import datetime

path='req'
images=[]
class_names=[]
list_req=os.listdir(path)
for cl in list_req:
    cur_image= cv2.imread(f'{path}/{cl}')
    images.append(cur_image)
    class_names.append(os.path.splitext(cl)[0])

def get_encodings(images):
    encode_list=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings=face_recognition.face_encodings(img)[0]
        encode_list.append(encodings)
    return encode_list

def mark_attendence(name):
    with open('attendence.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
             now = datetime.now()
             dtsring = now.strftime('%c')
             f.writelines(f'\n{name},{dtsring}')


encodelist_known=get_encodings(images)
print('Encoding Done')

cap= cv2.VideoCapture(0)

while True:

    success,img = cap.read()
    imgs= cv2.resize(img,(0,0),None,0.25,0.25)
    imgs= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    face_loc_cur= face_recognition.face_locations(imgs)
    face_encodings_cur= face_recognition.face_encodings(imgs,face_loc_cur)

    for encoding,faces_loc in zip(face_encodings_cur,face_loc_cur):
        matches=face_recognition.compare_faces(encodelist_known,encoding)
        face_dist=face_recognition.face_distance(encodelist_known,encoding)
        print(face_dist)
        matchIndex = np.argmin(face_dist)
    
        if matches[matchIndex]:
            name= class_names[matchIndex].upper()
            print(name)
            top, right, bottom, left=faces_loc
            #top, right, bottom, left=top*4,right*4,bottom*4,left*4
            cv2.rectangle(img,(left,top),(bottom,right),(255,255,0),2)
            cv2.rectangle(img,(top,right-35),(top,right),(255,255,2),cv2.FILLED)
            cv2.putText(img,name,(left+6,bottom-6),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1, cv2.LINE_AA)
            mark_attendence(name)
        

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

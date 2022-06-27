from unittest import result
import numpy as np
import cv2
import face_recognition
from scipy.misc import face
from torch import BenchmarkConfig
import matplotlib.pyplot as plt

imgben=face_recognition.load_image_file('ben/ben2.jpg')
imgben=cv2.cvtColor(imgben, cv2.COLOR_BGR2RGB)

imgben3=face_recognition.load_image_file('ben/ben8.jpg')
imgben3=cv2.cvtColor(imgben3, cv2.COLOR_BGR2RGB)

imgben4=face_recognition.load_image_file('ben/ben12.jpg')
imgben4=cv2.cvtColor(imgben4, cv2.COLOR_BGR2RGB)

imgbentest=face_recognition.load_image_file('ben/ben14.jpg')
imgbentest=cv2.cvtColor(imgbentest, cv2.COLOR_BGR2RGB)

face_loc= face_recognition.face_locations(imgben)[0]
encodings=face_recognition.face_encodings(imgben)[0]
cv2.rectangle(imgben,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(155,155,0),2)

face_loc3= face_recognition.face_locations(imgben3)[0]
encodings3=face_recognition.face_encodings(imgben3)[0]
cv2.rectangle(imgben3,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(155,155,0),2)

face_loc4= face_recognition.face_locations(imgben4)[0]
encodings4=face_recognition.face_encodings(imgben4)[0]
cv2.rectangle(imgben4,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(155,155,0),2)


face_loc_test= face_recognition.face_locations(imgbentest)[0]
encodings_test=face_recognition.face_encodings(imgbentest)[0]
cv2.rectangle(imgbentest,(face_loc_test[3],face_loc_test[0]),(face_loc_test[1],face_loc_test[2]),(155,155,0),2)


#cv2.imshow('ben2.jpg',imgben)
#cv2.imshow('ben8.jpg',imgben3)
#cv2.imshow('ben12.jpg',imgben4)
cv2.imshow('ben14.jpg',imgbentest)
results= face_recognition.compare_faces([encodings,encodings3,encodings4],encodings_test, tolerance=0.6)
dist=face_recognition.face_distance([encodings,encodings3,encodings4],encodings_test)

cv2.putText(imgbentest,f'{results[0]}{results[1]}{results[2]}{round(dist[0],2)}{round(dist[1],2)}{round(dist[2],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(155,155,0),2)
cv2.imshow('ben14.jpg',imgbentest)
print(results,dist)
cv2.waitKey(0)
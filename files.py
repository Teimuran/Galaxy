# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:41:06 2021

@author: IQ
"""

import os
import dlib
import cv2
from scipy.spatial import distance



stars = []
d=[]
dets=[]

def getDescriptor(img):
    descriptors =[]
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    dets=detector(img, 1)
    for num, face in enumerate(dets):
        shape=sp(img, face)
        face_descriptor = facerec.compute_face_descriptor(img,shape)
        descriptors.append(face_descriptor)
    return descriptors    
        
os.chdir('C:/galaxy')

predictor_path = 'shape_predictor_5_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

os.chdir('C:/galaxy') 

img = cv2.imread('7.jpg')
desk = getDescriptor(img)

# print(desk)

os.chdir('C:/galaxy/dataset')
stars = []
for _stars in os.listdir():
    print(os.getcwd())
    os.chdir(_stars)
    for name in os.listdir():
        print(os.getcwd()+'/'+name)
        img = cv2.imread(name)
        stars.append(getDescriptor(img))
        
    os.chdir('..')
     
print(len(stars))


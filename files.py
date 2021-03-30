# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:41:06 2021

@author: IQ
"""


import os
import dlib
import cv2
from scipy.spatial import distance  #импорт

stars = []
d=[]            #создание пустых массивов
dets=[]

def getDescriptor(img):
    descriptors =[]   #создание пустого массива
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   #преобразование из GBR в RGB 
    
    dets=detector(img, 1)                
    for num, face in enumerate(dets):      #цикл for где записываются все координаты лиц
    
        shape=sp(img, face)
        
        face_descriptor = facerec.compute_face_descriptor(img,shape)
        
        descriptors.append(face_descriptor)
    
    return descriptors    
        
os.chdir('C:/galaxy')

predictor_path = 'shape_predictor_5_face_landmarks.dat'   #импорт моделей распознавания
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()      #создание переменных с моделями распознавания
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

###############################################################

os.chdir('C:/galaxy/dataset')
stars = []    #создание пустого массива

for _stars in os.listdir():
    print(os.getcwd())

    os.chdir(_stars)  
    stars.append((_stars))

    pic = []   #создание пустого массива

    for name in os.listdir():         #цикл for с выводом полного пути файла
        print(os.getcwd()+'\ '+name)

        img = cv2.imread(name)

        pic.append(getDescriptor(img))   #получение дескрипторов с каждого фото из датасета

    stars.append(pic)    
    os.chdir('..')
     

maximum = 1

os.chdir('C:/galaxy')

w=cv2.imread('123.jpg')
pos = 0

desc = getDescriptor(w)

for pos, star in enumerate(stars):  #цикл сравнения с датасетом и выбором лучшего результата

    if pos % 2 == 1:

        for pic in star:
            delta = distance.euclidean(desc, pic)
            print(delta)
            res = delta

            if res < maximum:
                maximum = res
                ind = pos
            else:
                ind = ind
print(maximum, stars[ind-1])




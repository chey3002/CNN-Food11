# -*- coding: utf-8 -*-

"""Entrenar"""

import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'
"""
Parametros
"""
epocas=20 #Numero de iteraciones sobre el set de datos durante el entrnamiento
longitud, altura = 150, 150 #tamaño de imagen para el entrenamiento
batch_size = 32 #numero de imagenes por paso
pasos = 1000 #numero de veces que se procesara la información en cada éppoca
validation_steps = 300 #Al final de las épocas se realizara una validacion de 300 pasos
filtrosConv1 = 32 #Numero de filtros aplicados por combolución (Profundidad)
filtrosConv2 = 64 #
tamano_filtro1 = (3, 3) #tamaño del filtro usado en cada combolución
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2) #tamaño del filtro de max pooling
clases = 3 #numero de clases
lr = 0.0004 #coeficiente de aprendizaje


##Preparamos nuestras imagenes
#preprocesamiento
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255, #reescalar las imágenes rango de pixeles de 0 a 1
    shear_range=0.2, #Generar imagenes inclinadas
    zoom_range=0.2,  #posibilidad de hacer zoom
    horizontal_flip=True)#invertir imagen 
test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')
#Red neuronal combolucional
cnn = Sequential()#red secuencial
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu')) #creacion de la primera capa 
cnn.add(MaxPooling2D(pool_size=tamano_pool))#capa de pooling

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))#segunda caoa combolucional
cnn.add(MaxPooling2D(pool_size=tamano_pool))#segunda capa de pooling
#inicia la clasificación
cnn.add(Flatten())#transformar la información a una sola dimencion 
cnn.add(Dense(256, activation='relu'))#256 neuronas, con activación relu
cnn.add(Dropout(0.5))#Se apagaran el 50% de las neuronas para evitar sobreajuste, para adaptarse a informacion nueva
cnn.add(Dense(clases, activation='softmax'))#realiza la clasificación

cnn.compile(loss='categorical_crossentropy',#parametos que se utilizaran para el algoritmo
            optimizer=optimizers.Adam(lr=lr),#optimizador adam
            metrics=['accuracy'])#metrica utilizada

#entrenamiento
cnn.fit_generator(
    entrenamiento_generador,#imagenes de entrenamiento
    steps_per_epoch=pasos,#numero de pasos
    epochs=epocas,#numero de pasos
    validation_data=validacion_generador,#imagenes de validazion
    validation_steps=validation_steps)#pasos de validación
#guardar modelo como archivo
target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')

"""predict"""

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 150, 150#tamaño de la imagen
modelo = './modelo/modelo.h5'#direccion del modelo
pesos_modelo = './modelo/pesos.h5'#direccion de los pesos
cnn = load_model(modelo)#cargar modelo
cnn.load_weights(pesos_modelo)#cargar pesos
#funcion de prediccion
def predict(file):
  x = load_img(file, target_size=(longitud, altura))#cargar la imagen
  x = img_to_array(x)#tansformar imagen a arreglo
  x = np.expand_dims(x, axis=0)#en el eje 0 se agrega una dimención extra para procesar la información sin problema
  array = cnn.predict(x)#se llama a la red para realizar la predicción
  result = array[0]#obtenemos el resultado
  answer = np.argmax(result)#nos entrega el indice del valor mas alto
  #clasificación del resultado
  if answer == 0:
    print("pred: Perro")
  elif answer == 1:
    print("pred: Gato")
  elif answer == 2:
    print("pred: Gorila")

  return answer

file='./test/cat.jpg'
predict(file)#ingresar direccion de la imagen
file='./test/dog.jpg'
predict(file)#ingresar direccion de la imagen
file='./test/gorilla2.jpg'
predict(file)#ingresar direccion de la imagen

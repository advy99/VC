#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################

# En caso de necesitar instalar keras en google colab,
# ejecutar la siguiente línea:
# !pip install -q keras
# Importar librerías necesarias
import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.utils as np_utils

# Importar modelos y capas que se van a usar
# A completar
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten

# Importar el optimizador a usar
from keras.optimizers import SGD

# Importar el conjunto de datos
from keras.datasets import cifar100

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

# A esta función solo se la llama una vez. Devuelve 4
# vectores conteniendo, por este orden, las imágenes
# de entrenamiento, las clases de las imágenes de
# entrenamiento, las imágenes del conjunto de test y
# las clases del conjunto de test.
def cargarImagenes():
    # Cargamos Cifar100. Cada imagen tiene tamaño
    # (32 , 32, 3). Nos vamos a quedar con las
    # imágenes de 25 de las clases.
    (x_train, y_train), (x_test, y_test) = cifar100.load_data (label_mode ='fine')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    train_idx = np.isin(y_train, np.arange(25))
    train_idx = np.reshape (train_idx, -1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    test_idx = np.isin(y_test, np.arange(25))
    test_idx = np.reshape(test_idx, -1)
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    # Transformamos los vectores de clases en matrices.
    # Cada componente se convierte en un vector de ceros
    # con un uno en la componente correspondiente a la
    # clase a la que pertenece la imagen. Este paso es
    # necesario para la clasificación multiclase en keras.
    y_train = np_utils.to_categorical(y_train, 25)
    y_test = np_utils.to_categorical(y_test, 25)

    return x_train , y_train , x_test , y_test

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve la accuracy de un modelo,
# definida como el porcentaje de etiquetas bien predichas
# frente al total de etiquetas. Como parámetros es
# necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de
# keras (matrices donde cada etiqueta ocupa una fila,
# con un 1 en la posición de la clase a la que pertenece y un 0 en las demás).
def calcularAccuracy(labels, preds):
    labels = np.argmax(labels, axis = 1)
    preds = np.argmax(preds, axis = 1)
    accuracy = sum(labels == preds)/len(labels)
    return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución
# de la función de pérdida en el conjunto de train y
# en el de validación, y otra con la evolución de la
# accuracy en el conjunto de train y el de validación.
# Es necesario pasarle como parámetro el historial del
# entrenamiento del modelo (lo que devuelven las
# funciones fit() y fit_generator()).
def mostrarEvolucion(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()

    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy','Validation accuracy'])
    plt.show()

#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################

# A completar

# forma de las imagenes, como nos dice el guion
forma_entrada = (32, 32, 3)

modelo = Sequential()
modelo.add( Conv2D(6, kernel_size = (5,5), padding = "valid", input_shape = forma_entrada ) )
modelo.add( Activation("relu") )
modelo.add( MaxPooling2D(pool_size = (2, 2) ) )
modelo.add( Conv2D(16, kernel_size = (5,5), padding = "valid" ) )
modelo.add( Activation("relu") )
modelo.add( MaxPooling2D(pool_size = (2, 2) ) )
modelo.add( Flatten() )
modelo.add( Dense(units = 50) )
modelo.add( Activation("relu") )
modelo.add( Dense(units = 25) )
# es necesaria una activación softmax para transformar la salida
modelo.add( Activation("softmax") )

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

# A completar
optimizador = SGD()

# es multiclase, luego usamos categorical_crossentropy como perdida
modelo.compile( loss = keras.losses.categorical_crossentropy, optimizer = optimizador, metrics = ["accuracy"] )

# Una vez tenemos el modelo base, y antes de entrenar, vamos a guardar los
# pesos aleatorios con los que empieza la red, para poder reestablecerlos
# después y comparar resultados entre no usar mejoras y sí usarlas.
pesos_iniciales = modelo.get_weights()

# para ver como ha quedado el modelo
print( modelo.summary() )

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

# A completar
x_train, y_train, x_test, y_test = cargarImagenes()

# valor por defecto para batch_size en keras al hacer
tam_batch = 32

# porcentaje de entrenamiento que utilizará como validación
porcentaje_validacion = 0.1

# número de epocas a entrenar el modelo
epocas = 40

# entrenamos el modelo. Utilizamos verbose para ver una barra de progreso
evolucion_entrenamiento = modelo.fit(x_train, y_train, batch_size = tam_batch, epochs = epocas, validation_split = porcentaje_validacion, verbose = 1)

# mostramos la evolución del modelo tras entrenarlo
mostrarEvolucion(evolucion_entrenamiento)

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

# A completar
prediccion = modelo.predict(x_test, batch_size = tam_batch, verbose = 1)


precision_test = calcularAccuracy(y_test, prediccion)
print("Accuracy en el conjunto test: {}".format(precision_test))

#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

# A completar. Tanto la normalización de los datos como el data
# augmentation debe hacerse con la clase ImageDataGenerator.
# Se recomienda ir entrenando con cada paso para comprobar
# en qué grado mejora cada uno de ellos.

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
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Importar el optimizador a usar
from keras.optimizers import SGD, Adam, RMSprop

# Importar el conjunto de datos
from keras.datasets import cifar100


SEED = 1

# establecemos todas las semillas aleatorias al mismo valor
import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

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

x_train, y_train, x_test, y_test = cargarImagenes()

# valor por defecto para batch_size en keras al hacer
tam_batch = 32

# porcentaje de entrenamiento que utilizará como validación
porcentaje_validacion = 0.1

# número de epocas a entrenar el modelo
epocas = 30

# A completar

print("Ejercicio 1: Creación del modelo")

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

input("------------- Pulsa cualquier tecla para continuar -------------------")

#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

print("Apartado 2: Mejora del modelo")

# A completar. Tanto la normalización de los datos como el data
# augmentation debe hacerse con la clase ImageDataGenerator.
# Se recomienda ir entrenando con cada paso para comprobar
# en qué grado mejora cada uno de ellos.

print("Apartado 2: Normalización de datos")

generador_image_data = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True, validation_split = 0.1)
generador_image_data.fit(x_train)

modelo.set_weights(pesos_iniciales)


evolucion_entrenamiento_normalizado = modelo.fit_generator(
    generador_image_data.flow(x_train, y_train, batch_size = tam_batch, subset = 'training'),
    validation_data = generador_image_data.flow(x_train, y_train, batch_size = tam_batch, subset = 'validation'),
    steps_per_epoch = len(x_train) * (1.0 - porcentaje_validacion) / tam_batch,
    epochs = epocas,
    validation_steps = len(x_train) * porcentaje_validacion / tam_batch
)


mostrarEvolucion(evolucion_entrenamiento_normalizado)

# sin validacion para el test
generador_image_data_test = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True)
generador_image_data_test.fit(x_test)

prediccion_normalizada = modelo.predict_generator(generador_image_data_test.flow(x_test, batch_size = 1, shuffle = False), steps = len(x_test), verbose = 1)


precision_test_normalizado = calcularAccuracy(y_test, prediccion_normalizada)
print("Accuracy en el conjunto test con normalización: {}".format(precision_test_normalizado))


input("------------- Pulsa cualquier tecla para continuar -------------------")




print("Apartado 2: Aumento de datos")

generador_image_data_aumento = ImageDataGenerator(featurewise_center = True,
                                                  featurewise_std_normalization = True,
                                                  horizontal_flip = True,
                                                  validation_split = porcentaje_validacion)
generador_image_data_aumento.fit(x_train)

generador_image_data_aumento_val = ImageDataGenerator(featurewise_center = True,
                                                      featurewise_std_normalization = True,
                                                      validation_split= porcentaje_validacion)
generador_image_data_aumento_val.fit(x_train)

modelo.set_weights(pesos_iniciales)


evolucion_entrenamiento_norm_aumento = modelo.fit_generator(
    generador_image_data_aumento.flow(x_train, y_train, batch_size = tam_batch, subset = 'training'),
    validation_data = generador_image_data_aumento_val.flow(x_train, y_train, batch_size = tam_batch, subset = 'validation'),
    steps_per_epoch = len(x_train) * (1.0 - porcentaje_validacion) / tam_batch,
    epochs = epocas,
    validation_steps = len(x_train) * porcentaje_validacion / tam_batch
)


mostrarEvolucion(evolucion_entrenamiento_norm_aumento)

# sin validacion para el test
generador_image_data_test = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True)
generador_image_data_test.fit(x_test)

prediccion_norm_aumento = modelo.predict_generator(generador_image_data_test.flow(x_test, batch_size = 1, shuffle = False), steps = len(x_test), verbose = 1)


precision_test_norm_aumento = calcularAccuracy(y_test, prediccion_norm_aumento)
print("Accuracy en el conjunto test con normalizacion y aumento: {}".format(precision_test_norm_aumento))


input("------------- Pulsa cualquier tecla para continuar -------------------")

print("Ejercicio 2: Con BatchNormalization")

# forma de las imagenes, como nos dice el guion
forma_entrada = (32, 32, 3)

modelo_batchnormalization = Sequential()
modelo_batchnormalization.add( Conv2D(6, kernel_size = (5,5), padding = "same", input_shape = forma_entrada ) )
modelo_batchnormalization.add( BatchNormalization(renorm = True) )
modelo_batchnormalization.add( Activation("relu") )
modelo_batchnormalization.add( MaxPooling2D(pool_size = (2, 2) ) )
modelo_batchnormalization.add( Conv2D(16, kernel_size = (5,5), padding = "same" ) )
modelo_batchnormalization.add( BatchNormalization(renorm = True) )
modelo_batchnormalization.add( Activation("relu") )
modelo_batchnormalization.add( Dropout(0.2) )
modelo_batchnormalization.add( MaxPooling2D(pool_size = (2, 2) ) )
modelo_batchnormalization.add( Flatten() )
modelo_batchnormalization.add( Dense(units = 50) )
modelo_batchnormalization.add( BatchNormalization(renorm = True) )
modelo_batchnormalization.add( Activation("relu") )
modelo_batchnormalization.add( Dropout(0.1) )
modelo_batchnormalization.add( Dense(units = 25) )
# es necesaria una activación softmax para transformar la salida
modelo_batchnormalization.add( Activation("softmax") )

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

# A completar
optimizador = SGD()

# es multiclase, luego usamos categorical_crossentropy como perdida
modelo_batchnormalization.compile( loss = keras.losses.categorical_crossentropy, optimizer = optimizador, metrics = ["accuracy"] )

generador_image_data_aumento_batch = ImageDataGenerator(featurewise_center = True,
                                                  featurewise_std_normalization = True,
                                                  horizontal_flip = True,
                                                  validation_split = porcentaje_validacion)
generador_image_data_aumento_batch.fit(x_train)

generador_image_data_aumento_val_batch = ImageDataGenerator(featurewise_center = True,
                                                      featurewise_std_normalization = True,
                                                      validation_split= porcentaje_validacion)
generador_image_data_aumento_val_batch.fit(x_train)



evolucion_entrenamiento_norm_aumento_bach = modelo_batchnormalization.fit_generator(
    generador_image_data_aumento_batch.flow(x_train, y_train, batch_size = tam_batch, subset = 'training'),
    validation_data = generador_image_data_aumento_val_batch.flow(x_train, y_train, batch_size = tam_batch, subset = 'validation'),
    steps_per_epoch = len(x_train) * (1.0 - porcentaje_validacion) / tam_batch,
    epochs = epocas,
    validation_steps = len(x_train) * porcentaje_validacion / tam_batch
)


mostrarEvolucion(evolucion_entrenamiento_norm_aumento_bach)

# sin validacion para el test
generador_image_data_test_batch = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True)
generador_image_data_test_batch.fit(x_test)

prediccion_norm_aumento_batch = modelo_batchnormalization.predict_generator(generador_image_data_test_batch.flow(x_test, batch_size = 1, shuffle = False), steps = len(x_test), verbose = 1)


precision_test_norm_aumento_batch = calcularAccuracy(y_test, prediccion_norm_aumento_batch)
print("Accuracy en el conjunto test con normalizacion, aumento y batch normalization: {}".format(precision_test_norm_aumento_batch))



input("------------- Pulsa cualquier tecla para continuar -------------------")

print("Ejercicio 2: Con Dropout")

# forma de las imagenes, como nos dice el guion
forma_entrada = (32, 32, 3)

modelo_dropout = Sequential()
modelo_dropout.add( Conv2D(6, kernel_size = (5,5), padding = "same", input_shape = forma_entrada ) )
modelo_dropout.add( BatchNormalization(renorm = True) )
modelo_dropout.add( Activation("relu") )
modelo_dropout.add( MaxPooling2D(pool_size = (2, 2) ) )
modelo_dropout.add( Dropout(0.1) )
modelo_dropout.add( Conv2D(16, kernel_size = (5,5), padding = "same" ) )
modelo_dropout.add( BatchNormalization(renorm = True) )
modelo_dropout.add( Activation("relu") )
modelo_dropout.add( MaxPooling2D(pool_size = (2, 2) ) )
modelo_dropout.add( Flatten() )
modelo_dropout.add( Dropout(0.1) )
modelo_dropout.add( Dense(units = 50) )
modelo_dropout.add( BatchNormalization(renorm = True) )
modelo_dropout.add( Activation("relu") )
modelo_dropout.add( Dropout(0.05) )
modelo_dropout.add( Dense(units = 25) )
# es necesaria una activación softmax para transformar la salida
modelo_dropout.add( Activation("softmax") )

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

# A completar
optimizador = SGD()

# es multiclase, luego usamos categorical_crossentropy como perdida
modelo_dropout.compile( loss = keras.losses.categorical_crossentropy, optimizer = optimizador, metrics = ["accuracy"] )

generador_image_data_dropout = ImageDataGenerator(featurewise_center = True,
                                                  featurewise_std_normalization = True,
                                                  horizontal_flip = True,
                                                  validation_split = porcentaje_validacion)
generador_image_data_dropout.fit(x_train)

generador_image_data_val_dropout = ImageDataGenerator(featurewise_center = True,
                                                      featurewise_std_normalization = True,
                                                      validation_split= porcentaje_validacion)
generador_image_data_val_dropout.fit(x_train)



evolucion_entrenamiento_dropout = modelo_dropout.fit_generator(
    generador_image_data_dropout.flow(x_train, y_train, batch_size = tam_batch, subset = 'training'),
    validation_data = generador_image_data_val_dropout.flow(x_train, y_train, batch_size = tam_batch, subset = 'validation'),
    steps_per_epoch = len(x_train) * (1.0 - porcentaje_validacion) / tam_batch,
    epochs = epocas,
    validation_steps = len(x_train) * porcentaje_validacion / tam_batch
)


mostrarEvolucion(evolucion_entrenamiento_dropout)

# sin validacion para el test
generador_image_data_test_dropout = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True)
generador_image_data_test_dropout.fit(x_test)

prediccion_norm_dropout = modelo_dropout.predict_generator(generador_image_data_test_dropout.flow(x_test, batch_size = 1, shuffle = False), steps = len(x_test), verbose = 1)


precision_test_norm_dropout = calcularAccuracy(y_test, prediccion_norm_dropout)
print("Accuracy en el conjunto test con normalizacion, aumento, batch normalization y dropout: {}".format(precision_test_norm_dropout))



input("------------- Pulsa cualquier tecla para continuar -------------------")

print("Ejercicio Bonus: ELU, Adam como optimizador, nuevo modelo")

# forma de las imagenes, como nos dice el guion
forma_entrada = (32, 32, 3)

modelo_bonus = Sequential()
modelo_bonus.add( Conv2D(32, kernel_size=(3, 3), input_shape=forma_entrada))
modelo_bonus.add( Activation("elu") )
modelo_bonus.add( BatchNormalization(renorm = True) )
modelo_bonus.add( MaxPooling2D(pool_size=(2, 2)))
modelo_bonus.add( Conv2D(64, kernel_size=(3, 3)) )
modelo_bonus.add( Activation("elu") )
modelo_bonus.add( BatchNormalization(renorm = True) )
modelo_bonus.add( MaxPooling2D(pool_size=(2, 2)))
modelo_bonus.add( Conv2D(128, kernel_size=(3, 3)))
modelo_bonus.add( Activation("elu") )
modelo_bonus.add( BatchNormalization(renorm = True) )
modelo_bonus.add( MaxPooling2D(pool_size=(2, 2)))
modelo_bonus.add( Flatten())
modelo_bonus.add( Dense(256, activation='elu'))
modelo_bonus.add( Dense(128, activation='elu'))
modelo_bonus.add( Dense(25, activation='softmax'))




#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

# A completar
optimizador = Adam()

# es multiclase, luego usamos categorical_crossentropy como perdida
modelo_bonus.compile( loss = keras.losses.categorical_crossentropy, optimizer = optimizador, metrics = ["accuracy"] )

print(modelo_bonus.summary())

generador_image_data_bonus = ImageDataGenerator(featurewise_center = True,
                                                  featurewise_std_normalization = True,
                                                  horizontal_flip = True,
                                                  validation_split = porcentaje_validacion)
generador_image_data_bonus.fit(x_train)

generador_image_data_val_bonus = ImageDataGenerator(featurewise_center = True,
                                                      featurewise_std_normalization = True,
                                                      validation_split= porcentaje_validacion)
generador_image_data_val_bonus.fit(x_train)



evolucion_entrenamiento_bonus = modelo_bonus.fit_generator(
    generador_image_data_bonus.flow(x_train, y_train, batch_size = tam_batch, subset = 'training'),
    validation_data = generador_image_data_val_bonus.flow(x_train, y_train, batch_size = tam_batch, subset = 'validation'),
    steps_per_epoch = len(x_train) * (1.0 - porcentaje_validacion) / tam_batch,
    epochs = epocas,
    validation_steps = len(x_train) * porcentaje_validacion / tam_batch
)


mostrarEvolucion(evolucion_entrenamiento_bonus)

# sin validacion para el test
generador_image_data_test_bonus = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True)
generador_image_data_test_bonus.fit(x_test)

prediccion_norm_bonus = modelo_bonus.predict_generator(generador_image_data_test_bonus.flow(x_test, batch_size = 1, shuffle = False), steps = len(x_test), verbose = 1)


precision_test_norm_bonus = calcularAccuracy(y_test, prediccion_norm_bonus)
print("Accuracy en el conjunto test con normalizacion, aumento, batch normalization, elu, optimizador Adam y nuevo modelo: {}".format(precision_test_norm_bonus))




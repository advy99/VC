# -*- coding: utf-8 -*-
"""EsquemaParte3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ge_Pw-txgQ1PRJTdJPioBOl02yILeVuf
"""

#########################################################################
################### OBTENER LA BASE DE DATOS ############################
#########################################################################

# Descargar las imágenes de http://www.vision.caltech.edu/visipedia/CUB-200.html
# Descomprimir el fichero.
# Descargar también el fichero list.tar.gz, descomprimirlo y guardar los ficheros
# test.txt y train.txt dentro de la carpeta de imágenes anterior. Estos
# dos ficheros contienen la partición en train y test del conjunto de datos.

##### EN CASO DE USAR COLABORATORY
# Sube tanto las imágenes como los ficheros text.txt y train.txt a tu drive.
# Después, ejecuta esta celda y sigue las instrucciones para montar
# tu drive en colaboratory.
#from google.colab import drive
#drive.mount('/content/drive')

#########################################################################
################ CARGAR LAS LIBRERÍAS NECESARIAS ########################
#########################################################################

# Terminar de rellenar este bloque con lo que vaya haciendo falta

# Importar librerías necesarias
import numpy as np
import keras
import keras.utils as np_utils
from keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt

# Importar el optimizador a usar
from keras.optimizers import SGD

# Importar modelos y capas específicas que se van a usar

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D

# Importar el modelo ResNet50 y su respectiva función de preprocesamiento,
# que es necesario pasarle a las imágenes para usar este modelo
from keras.applications.resnet50 import ResNet50, preprocess_input


# Importar el optimizador a usar
from keras.optimizers import SGD

SEED = 1

# establecemos todas las semillas aleatorias al mismo valor
import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)



#########################################################################
################## FUNCIÓN PARA LEER LAS IMÁGENES #######################
#########################################################################

# Dado un fichero train.txt o test.txt y el path donde se encuentran los
# ficheros y las imágenes, esta función lee las imágenes
# especificadas en ese fichero y devuelve las imágenes en un vector y
# sus clases en otro.

def leerImagenes(vec_imagenes, path):
  clases = np.array([img.split('/')[0] for img in vec_imagenes])
  imagenes = np.array([img_to_array(load_img(path + "/" + img,
                                             target_size = (224, 224)))
                       for img in vec_imagenes])
  return imagenes, clases

#########################################################################
############# FUNCIÓN PARA CARGAR EL CONJUNTO DE DATOS ##################
#########################################################################

# Usando la función anterior, y dado el path donde se encuentran las
# imágenes y los archivos "train.txt" y "test.txt", devuelve las
# imágenes y las clases de train y test para usarlas con keras
# directamente.

def cargarDatos(path):
  # Cargamos los ficheros
  train_images = np.loadtxt(path + "/train.txt", dtype = str)
  test_images = np.loadtxt(path + "/test.txt", dtype = str)

  # Leemos las imágenes con la función anterior
  train, train_clases = leerImagenes(train_images, path)
  test, test_clases = leerImagenes(test_images, path)

  # Pasamos los vectores de las clases a matrices
  # Para ello, primero pasamos las clases a números enteros
  clases_posibles = np.unique(np.copy(train_clases))
  for i in range(len(clases_posibles)):
    train_clases[train_clases == clases_posibles[i]] = i
    test_clases[test_clases == clases_posibles[i]] = i

  # Después, usamos la función to_categorical()
  train_clases = np_utils.to_categorical(train_clases, 200)
  test_clases = np_utils.to_categorical(test_clases, 200)

  # Barajar los datos
  train_perm = np.random.permutation(len(train))
  train = train[train_perm]
  train_clases = train_clases[train_perm]

  test_perm = np.random.permutation(len(test))
  test = test[test_perm]
  test_clases = test_clases[test_perm]

  return train, train_clases, test, test_clases

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve el accuracy de un modelo, definido como el
# porcentaje de etiquetas bien predichas frente al total de etiquetas.
# Como parámetros es necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de keras (matrices
# donde cada etiqueta ocupa una fila, con un 1 en la posición de la clase
# a la que pertenece y 0 en las demás).

def calcularAccuracy(labels, preds):
  labels = np.argmax(labels, axis = 1)
  preds = np.argmax(preds, axis = 1)

  accuracy = sum(labels == preds)/len(labels)

  return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución de la función
# de pérdida en el conjunto de train y en el de validación, y otra
# con la evolución del accuracy en el conjunto de train y en el de
# validación. Es necesario pasarle como parámetro el historial
# del entrenamiento del modelo (lo que devuelven las funciones
# fit() y fit_generator()).

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
  plt.legend(['Training accuracy', 'Validation accuracy'])
  plt.show()

"""## Usar ResNet50 preentrenada en ImageNet como un extractor de características"""

# Definir un objeto de la clase ImageDataGenerator para train y otro para test
# con sus respectivos argumentos.
# A completar

# si se usa colab, cambiar a la ruta donde tenga las imagenes
#RUTA_IMAGENES = "/content/drive/My Drive/VC_p2/imagenes"
RUTA_IMAGENES = "imagenes"
x_train, y_train, x_test, y_test = cargarDatos(RUTA_IMAGENES)


tam_batch = 30
epocas = 10
porcentaje_val = 0.1
optimizador = SGD()

generador_datos_resnet50 = ImageDataGenerator(preprocessing_function = preprocess_input)

# Definir el modelo ResNet50 (preentrenado en ImageNet y sin la última capa).
# A completar
modelo_resnet50 = ResNet50(include_top = False, weights = "imagenet", pooling = "avg")

modelo_resnet50.trainable = False


# Extraer las características las imágenes con el modelo anterior.
# A completar

# extraemos las de entrenamiento
caracteristicas_train = modelo_resnet50.predict_generator(generador_datos_resnet50.flow(x_train, batch_size = 1, shuffle = False), steps = len(x_train), verbose = 1)

# extraemos las caracteristicas de test
caracteristicas_test = modelo_resnet50.predict_generator(generador_datos_resnet50.flow(x_test, batch_size = 1, shuffle = False), steps = len(x_test), verbose = 1)

# Las características extraídas en el paso anterior van a ser la entrada
# de un pequeño modelo de dos capas Fully Conected, donde la última será la que
# nos clasifique las clases de Caltech-UCSD (200 clases). De esta forma, es
# como si hubiéramos fijado todos los parámetros de ResNet50 y estuviésemos
# entrenando únicamente las capas añadidas. Definir dicho modelo.
# A completar: definición del modelo, del optimizador y compilación y
# entrenamiento del modelo.
# En la función fit() puedes usar el argumento validation_split

# creamos el modelo Fully Conected de dos capas
modelo_dos_capas_FC = Sequential()
modelo_dos_capas_FC.add(Dense(256, activation = "relu", input_shape = (2048,) ))
modelo_dos_capas_FC.add(Dense(200, activation = "softmax"))


# compilamos el modelo

modelo_dos_capas_FC.compile(loss = keras.losses.categorical_crossentropy, optimizer = optimizador, metrics = ["accuracy"])


# mostramos el resultado
print(modelo_dos_capas_FC.summary())


# guardamos los pesos iniciales por si reentrenamos
pesos_iniciales_FC = modelo_dos_capas_FC.get_weights()

# entrenamos el modelo
evolucion = modelo_dos_capas_FC.fit(caracteristicas_train, y_train, epochs = epocas, batch_size = tam_batch, validation_split = porcentaje_val, verbose = 1)


mostrarEvolucion(evolucion)

# predecimos los valores de test
prediccion_test = modelo_dos_capas_FC.predict(caracteristicas_test, batch_size = tam_batch, verbose = 1)

# obtenemos el accuracy
accuracy_test = calcularAccuracy(y_test, prediccion_test)

print("El accuracy en test es de: {}".format(accuracy_test))


print("Apartado 3.1.B:")


# con esto conseguimos el modelo sin la ultima capa de pooling
modelo_resnet50_sin_av_pooling = ResNet50(include_top = False, weights = "imagenet", pooling = None)



# Extraer las características las imágenes con el modelo anterior.
# A completar

# extraemos las de entrenamiento
caracteristicas_train_b = modelo_resnet50_sin_av_pooling.predict_generator(generador_datos_resnet50.flow(x_train, batch_size = 1, shuffle = False), steps = len(x_train), verbose = 1)

# extraemos las caracteristicas de test
caracteristicas_test_b = modelo_resnet50_sin_av_pooling.predict_generator(generador_datos_resnet50.flow(x_test, batch_size = 1, shuffle = False), steps = len(x_test), verbose = 1)

forma = caracteristicas_train_b.shape[1:]

nuevo_modelo = Sequential()
nuevo_modelo.add(Dropout(0.2, input_shape = forma ))
nuevo_modelo.add( Conv2D(250, kernel_size = (5,5), padding = "valid" ) )
nuevo_modelo.add( BatchNormalization(renorm = True) )
nuevo_modelo.add( Activation("relu") )
nuevo_modelo.add(Dropout(0.2) )
nuevo_modelo.add(GlobalAveragePooling2D())
nuevo_modelo.add(Dense(200, activation = "softmax"))


nuevo_modelo.compile(loss = keras.losses.categorical_crossentropy, optimizer = optimizador, metrics = ["accuracy"])

print(nuevo_modelo.summary())

evolucion_b = nuevo_modelo.fit(caracteristicas_train_b, y_train, epochs = epocas, batch_size = tam_batch, validation_split = porcentaje_val, verbose = 1)


mostrarEvolucion(evolucion_b)

# predecimos los valores de test
prediccion_test_b = nuevo_modelo.predict(caracteristicas_test_b, batch_size = tam_batch, verbose = 1)

# obtenemos el accuracy
accuracy_test_b = calcularAccuracy(y_test, prediccion_test_b)

print("El accuracy en test para el apartado B es de: {}".format(accuracy_test_b))



"""## Reentrenar ResNet50 (fine tunning)"""

# Definir un objeto de la clase ImageDataGenerator para train y otro para test
# con sus respectivos argumentos.
# A completar
generador_datos_train = ImageDataGenerator(validation_split = porcentaje_val, preprocessing_function = preprocess_input)


generador_datos_test = ImageDataGenerator(preprocessing_function = preprocess_input)


# Añadir nuevas capas al final de ResNet50 (recuerda que es una instancia de
# la clase Model).
salida_resnet = modelo_resnet50.output
salida_resnet = Dropout(0.2) (salida_resnet)
salida_resnet = BatchNormalization(renorm = True) (salida_resnet)
salida_resnet = Activation("relu") (salida_resnet)
salida_resnet = Dropout(0.2) (salida_resnet)
salida_resnet = Dense(200, activation = "softmax") (salida_resnet)


modelo = Model(inputs = modelo_resnet50.input, outputs = salida_resnet)

# Compilación y entrenamiento del modelo.
# A completar.
modelo.compile(loss = keras.losses.categorical_crossentropy, optimizer = optimizador, metrics = ["accuracy"])


print(modelo.summary())


# entrenamos con el modelo_resnet50 congelado, solo se entrenará la
# segunda parte
evolucion_fine = modelo.fit_generator(
    generador_datos_train.flow(x_train, y_train, batch_size = tam_batch, subset = "training"),
    epochs = epocas,
    validation_data = generador_datos_train.flow(x_train, y_train, batch_size = tam_batch, subset = "validation"),
    steps_per_epoch = len(x_train) * (1 - porcentaje_val)/tam_batch,
    validation_steps = len(x_train) * porcentaje_val / tam_batch
)

mostrarEvolucion(evolucion_fine)

predicciones_fine = modelo.predict_generator(generador_datos_test.flow(x_test, batch_size = 1, shuffle = False), steps = len(x_test), verbose = 1)

accuracy_fine = calcularAccuracy(y_test, predicciones_fine)
print("Accuracy con fine tuning (antes de entrenar todo el modelo): {}".format(accuracy_fine))

modelo_resnet50.trainable = True

modelo = Model(inputs = modelo_resnet50.input, outputs = salida_resnet)

modelo.compile(loss = keras.losses.categorical_crossentropy, optimizer = optimizador, metrics = ["accuracy"])

print(modelo.summary())

# entrenamos con el modelo_resnet50 congelado, solo se entrenará la
# segunda parte
evolucion_fine = modelo.fit(
    generador_datos_train.flow(x_train, y_train, batch_size = tam_batch, subset = "training"),
    epochs = epocas,
    validation_data = generador_datos_train.flow(x_train, y_train, batch_size = tam_batch, subset = "validation"),
    steps_per_epoch = len(x_train) * (1 - porcentaje_val)/tam_batch,
    validation_steps = len(x_train) * porcentaje_val / tam_batch
)

mostrarEvolucion(evolucion_fine)

predicciones_fine = modelo.predict_generator(generador_datos_test.flow(x_test, batch_size = 1, shuffle = False), steps = len(x_test), verbose = 1)

accuracy_fine = calcularAccuracy(y_test, predicciones_fine)
print("Accuracy con fine tuning: {}".format(accuracy_fine))




#!/usr/bin/env python3

"""

Visión por Computador: Práctica 1

Antonio David Villegas Yeguas

"""

# modulos utilizados a lo largo de la práctica
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

"""

Funciones básicas de la práctica 0 que reutilizaré

"""

def leeimagen(fichero, flag_color):

    imagen = cv.imread(fichero, flag_color)

    return imagen



def mostrar_imagen(imagen, titulo=""):
    plt.title(titulo)
    # si la imagen es tribanda, tenemos que invertir los canales B y R
    # si es en blanco y negro, tenemos que decirle a matplotlib que es monobanda
    if imagen.ndim == 3 and imagen.shape[2] >= 3:
        plt.imshow(imagen[:,:,::-1])
    else:
        plt.imshow(imagen, cmap='gray')
    plt.show()

def mostrar_imagenes(imagenes, titulos=None, titulo=""):
    plt.clf()
    plt.title(titulo)
    fig = plt.figure()
    ax = []

    columnas = len(imagenes)
    filas = 1

    for i in range(1, columnas*filas +1):
        ax.append(fig.add_subplot(filas, columnas, i))

        if titulos != None:
            ax[-1].set_title(titulos[i-1])

        if imagenes[i-1].ndim == 3 and imagenes[i-1].shape[2] >= 3:
            plt.imshow(imagenes[i-1][:,:,::-1])
        else:
            plt.imshow(imagenes[i-1], cmap='gray')

    plt.show()


def normaliza_imagen(imagen):

    # trabajamos en una copia de la imagen
    img_normalizada = np.copy(imagen)

    # sacamos el minimo y el maximo de todos los valores
    minimo = np.min(img_normalizada)
    maximo = np.max(img_normalizada)

    # si el numero es negativo, al hacer - (-minimo), lo va a sumar hasta llegar 0
    # luego no hay perdida de información
    if maximo - minimo != 0:
        img_normalizada = (img_normalizada - minimo) / (maximo - minimo)
    else:
        # en caso de que sean todos iguales, interpretamos que la imagen es todo
        # negro
        img_normalizada = img_normalizada - minimo

    # al hacerle la operación de forma matricial, no hay que tener en cuenta si
    # es monobanda o es tribanda

    return img_normalizada

"""

Ejercicio 1

Apartado A

"""

def funcion_gaussiana(x, sigma):
    """
    Evaluar la función gausiana en un punto concreto x con un sigma dado

    """

    return math.exp(- (x**2) / (2 * sigma**2) )

def derivada_f_gaussiana(x, sigma):
    """
    Primera derivada de la función gaussiana en un x concreto con un sigma dado
    """

    return ( - (math.exp(- (x**2)/(2 * sigma**2)  ) * x ) / (sigma**2) )

def segunda_derivada_f_gaussiana(x, sigma):
    """
    Segunda derivada de la función gaussiana en un x concreto con un sigma dado
    """

    return (- ( (-x**2 + sigma**2) / ( math.exp((x**2) / (2*sigma**2)) * sigma**4 ) ) )

def kernel_gaussiano_1d(sigma=None, func=funcion_gaussiana, tam_mascara=None):

    parametros_correctos = tam_mascara != None or sigma != None

    if tam_mascara != None:
        sigma = (tam_mascara - 1) / 6
    elif sigma != None:
        # usaremos un intervalo de  para obtener el kernel
        tam_mascara = 2 * 3 * sigma + 1
    else:
        print("ERROR: Debes aportar un sigma o un tamaño de máscara")

    # inicializamos kernels vacios
    kernel = []
    kernel_normalizado = []

    if parametros_correctos:

        mitad_intervalo = np.floor(tam_mascara/2)
        mitad_intervalo = int(mitad_intervalo)

        # usamos la mitad ya que si tam_mascara vale x, en realidad vamos desde
        # -x/2 .. x/2, teniendo el total un tamaño de x
        for x in range(-mitad_intervalo, mitad_intervalo + 1):
            kernel.append( func(x, sigma) )

        kernel_normalizado = kernel

        # si es la funcion gaussiana normalizamos, si es alguna de sus derivadas
        # no tenemos que normalizar
        if func == funcion_gaussiana:
            kernel_normalizado = kernel_normalizado / np.sum(kernel)

    return kernel_normalizado


mascara_gaussiana_sigma_1 = kernel_gaussiano_1d(1)
mascara_primera_deriv_t_5 = kernel_gaussiano_1d(None, derivada_f_gaussiana, 5)
mascara_segunda_deriv_t_7 = kernel_gaussiano_1d(None, derivada_f_gaussiana, 7)

print("Máscara obtenida de la función gaussiana con sigma = 1: ")
print(mascara_gaussiana_sigma_1)

print("\nMáscara obtenida de la función derivada de la gaussiana con tam. mascara = 5: ")
print(mascara_primera_deriv_t_5)

print("\nMáscara obtenida de la función segunda derivada de la gaussiana con tam. mascara = 7: ")
print(mascara_segunda_deriv_t_7)


input("\n---------- Pulsa una tecla para continuar ----------\n")



"""
Ejercicio 1

Apartado B
"""


def aplicar_convolucion(imagen, mascara_horizontal, mascara_vertical):

    # calculamos el tamaño que tendran los bordes
    borde = int( (len(mascara_horizontal) - 1)/2 )

    imagen_con_bordes = cv.copyMakeBorder(imagen, borde, borde, borde, borde, cv.BORDER_REFLECT_101)

    anchura = imagen_con_bordes.shape[0]
    altura = imagen_con_bordes.shape[1]

    imagen_modificada = np.zeros(imagen.shape)

    # recorremos la imagen
    for x in range(borde, anchura - borde):
        for y in range(borde, altura - borde):
            # multiplicamos la mascara que queremos que se aplique de forma horizontal
            # con los elementos de distintas columnas, pero mismas filas
            imagen_modificada[x-borde, y-borde] = np.dot(mascara_horizontal, imagen_con_bordes[x, y-borde:y+borde + 1])

    imagen_con_bordes = cv.copyMakeBorder(imagen_modificada, borde, borde, borde, borde, cv.BORDER_REFLECT_101)

    for x in range(borde, anchura - borde):
        for y in range(borde, altura - borde):
            # multiplicamos la mascara que queremos que se aplique de forma vertical
            # con los elementos de distintas filas, pero misma columna
            imagen_modificada[x-borde, y-borde] = np.dot(mascara_vertical, imagen_con_bordes[x-borde:x+borde+1, y].T)

    # devolvemos la imagen
    return imagen_modificada

bicycle = leeimagen("imagenes/bicycle.bmp", 0)
bird = leeimagen("imagenes/bird.bmp", 0)
cat = leeimagen("imagenes/cat.bmp", 0)
dog = leeimagen("imagenes/dog.bmp", 0)
einstein = leeimagen("imagenes/einstein.bmp", 0)
fish = leeimagen("imagenes/fish.bmp", 0)
marilyn = leeimagen("imagenes/marilyn.bmp", 0)
motorcycle = leeimagen("imagenes/motorcycle.bmp", 0)
plane = leeimagen("imagenes/plane.bmp", 0)
submarine = leeimagen("imagenes/submarine.bmp", 0)

bicycle = normaliza_imagen(bicycle)

mascara = kernel_gaussiano_1d(tam_mascara=9)

nueva_imagen = aplicar_convolucion(bicycle, mascara, mascara)

#
#
# imagen_cv = aplicar_convolucion(bicycle, kernel1, kernel2)

# mostrar_imagen(bicycle)
#
# mostrar_imagen(nueva_imagen)
#
# mostrar_imagen(imagen_cv)

imagen_cv = cv.GaussianBlur(bicycle, ksize=(9,9), sigmaX=-1, sigmaY=-1)

titulos = ["Original", "Convolución con máscara gaussiana propia", "Con cv.GaussianBlur"]
mostrar_imagenes([bicycle, nueva_imagen, imagen_cv], titulos)



"""
Ejercicio 1

Apartado C

"""


mascara_primera_derivada = kernel_gaussiano_1d(func=derivada_f_gaussiana, tam_mascara=15)
mascara_segunda_derivada = kernel_gaussiano_1d(func=segunda_derivada_f_gaussiana, tam_mascara=15)

kernel_15, basura = cv.getDerivKernels(1, 1, 15)

kernel_15 = kernel_15.reshape((15,))

kernel_15_2, basura = cv.getDerivKernels(2, 2, 15)

kernel_15_2 = kernel_15_2.reshape((15,))

plt.clf()
plt.title("Máscara primera derivada f. gaussiana propia, t. mascara = 15.")
plt.scatter(range(15), mascara_primera_derivada)
plt.show()

plt.clf()
plt.title("Máscara primera derivada f. gaussiana deriv kernels, t. mascara = 15.")
plt.scatter(range(15), kernel_15)
plt.show()

plt.clf()
plt.title("Máscara segunda derivada f. gaussiana propia, t.mascara = 15.")
plt.scatter(range(15), mascara_segunda_derivada)
plt.show()

plt.clf()
plt.title("Máscara segunda derivada f. gaussiana deriv kernels, t.mascara = 15.")
plt.scatter(range(15), kernel_15_2)
plt.show()


"""
Ejercicio 1

Apartado D

"""



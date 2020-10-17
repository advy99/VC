#!/usr/bin/env python3

"""

Visión por Computador: Práctica 1

Antonio David Villegas Yeguas

"""

# modulos utilizados a lo largo de la práctica
import cv2 as cv
import matplotlib as plt
import numpy as np

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

def kernel_gaussiano_1d(sigma, func=funcion_gaussiana):

    # usaremos un intervalo de [-3 * sigma, 3 * sigma] para obtener el kernel
    comienzo_intervalo = int(- (3 * sigma))
    fin_intervalo = -comienzo_intervalo

    kernel = []

    # usaremos fin_intervalo+1 para que fin_intervalo este incluido
    for x in range(comienzo_intervalo, fin_intervalo + 1):
        kernel.append( func(x, sigma) )

    # normalizamos el kernel dividiendo cada valor por la suma de todos
    # haciendo que en total sumen 1
    kernel_normalizado = kernel / np.sum(kernel)

    return kernel_normalizado


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




#!/usr/bin/env python3

"""

Visión por Computador: Práctica 1

Antonio David Villegas Yeguas

"""

# modulos utilizados a lo largo de la práctica
import cv2 as cv
import matplotlib as plt
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
        sigma = tam_mascara - 1 / 6
    elif sigma != None:
        # usaremos un intervalo de  para obtener el kernel
        tam_mascara = 2 * 3 * sigma + 1
    else:
        print("ERROR: Debes aportar un sigma o un tamaño de máscara")


    kernel = []
    kernel_normalizado = []

    if parametros_correctos:

        mitad_intervalo = np.floor(tam_mascara/2)
        mitad_intervalo = int(mitad_intervalo)

        # usamos la mitad ya que si tam_mascara vale x, en realidad vamos desde
        # -x/2 .. x/2, teniendo el total un tamaño de x
        for x in range(-mitad_intervalo, mitad_intervalo + 1):
            kernel.append( func(x, sigma) )

        # normalizamos el kernel dividiendo cada valor por la suma de todos
        # haciendo que en total sumen 1
        kernel_normalizado = kernel

        # si la suma de los valores del kernel no es 0 lo normalizamos
        # para que la suma sea 1
        # en lugar de comparar con 0, comparamos si el valor absoluto es mayor
        # que cierto epsilon ya que comprar con 0 puede tener problemas de coma
        # flotante y por lo tanto comportamientos inesperados
        if abs(np.sum(kernel)) > 0.005:
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




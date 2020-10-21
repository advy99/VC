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
    plt.title(titulo)
    fig = plt.figure()
    ax = []

    # no se porque mete una figura en blanco, la cerramos para que no se muestre
    plt.close(1)

    columnas = len(imagenes)
    filas = 1

    # para cada imagen, la añadimos
    for i in range(1, columnas*filas +1):
        ax.append(fig.add_subplot(filas, columnas, i))

        # le ponemos su titulo
        if titulos != None:
            ax[-1].set_title(titulos[i-1])

        # y la mostramos
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
    """
    Función para crear un kernel gaussiano 1D
    """

    # nos tienen que dar un sigma o un tamaño de mascara
    parametros_correctos = tam_mascara != None or sigma != None

    # si nos dan un tamaño de mascara, calculamos un sigma
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

    # si han ejecutado la funcion de forma correcta
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

# probamos la función

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


def aplicar_convolucion(imagen, mascara_horizontal, mascara_vertical, t_borde=cv.BORDER_REFLECT_101):
    """
    Función para aplicar una convolucion. Dada una imagen, la mascara a aplicar
    horizontalmente y verticalmente, y el tipo de borde

    """

    # calculamos el tamaño que tendran los bordes
    borde = int( (len(mascara_horizontal) - 1)/2 )

    # añadimos los bordes para aplicar la convolucion a toda la image
    imagen_con_bordes = cv.copyMakeBorder(imagen, borde, borde, borde, borde, t_borde)

    anchura = imagen_con_bordes.shape[0]
    altura = imagen_con_bordes.shape[1]

    imagen_modificada = np.zeros(imagen.shape)

    # recorremos la imagen
    for x in range(borde, anchura - borde):
        for y in range(borde, altura - borde):
            # multiplicamos la mascara que queremos que se aplique de forma horizontal
            # con los elementos de distintas columnas, pero mismas filas
            imagen_modificada[x-borde, y-borde] = np.dot(mascara_horizontal, imagen_con_bordes[x, y-borde:y+borde + 1])

    # como la imagen aplicada una convolucion la hemos ajustado, necesitamos volver
    # a añadir los bordes
    imagen_con_bordes = cv.copyMakeBorder(imagen_modificada, borde, borde, borde, borde, cv.BORDER_REFLECT_101)

    for x in range(borde, anchura - borde):
        for y in range(borde, altura - borde):
            # multiplicamos la mascara que queremos que se aplique de forma vertical
            # con los elementos de distintas filas, pero misma columna
            imagen_modificada[x-borde, y-borde] = np.dot(mascara_vertical, imagen_con_bordes[x-borde:x+borde+1, y].T)

    # devolvemos la imagen
    return normaliza_imagen(imagen_modificada)

# leemos todas las imagenes en B/N y las normalizamos
bicycle    = normaliza_imagen(leeimagen("imagenes/bicycle.bmp", 0))
bird       = normaliza_imagen(leeimagen("imagenes/bird.bmp", 0))
cat        = normaliza_imagen(leeimagen("imagenes/cat.bmp", 0))
dog        = normaliza_imagen(leeimagen("imagenes/dog.bmp", 0))
einstein   = normaliza_imagen(leeimagen("imagenes/einstein.bmp", 0))
fish       = normaliza_imagen(leeimagen("imagenes/fish.bmp", 0))
marilyn    = normaliza_imagen(leeimagen("imagenes/marilyn.bmp", 0))
motorcycle = normaliza_imagen(leeimagen("imagenes/motorcycle.bmp", 0))
plane      = normaliza_imagen(leeimagen("imagenes/plane.bmp", 0))
submarine  = normaliza_imagen(leeimagen("imagenes/submarine.bmp", 0))

# probamos a aplicar una convolucion y la mostramos
mascara = kernel_gaussiano_1d(tam_mascara=9)

nueva_imagen = aplicar_convolucion(bicycle, mascara, mascara)

# comparamos con GaussianBlur
imagen_cv = cv.GaussianBlur(bicycle, ksize=(9,9), sigmaX=-1, sigmaY=-1)

titulos = ["Original", "Convolución con máscara gaussiana propia", "Con cv.GaussianBlur"]
mostrar_imagenes([bicycle, nueva_imagen, imagen_cv], titulos)


input("\n---------- Pulsa una tecla para continuar ----------\n")

"""
Ejercicio 1

Apartado C

"""

# comparamos los kernels obtenidos con los kernels de getDerivKernels
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


input("\n---------- Pulsa una tecla para continuar ----------\n")

"""
Ejercicio 1

Apartado D

"""


def mascara_laplaciana(imagen, sigma=None, tam_mascara=None):
    """
    Función para aplicar una máscara laplaciana a una imagen
    """

    #si nos dan el sigma, buscamos el kernel de la primera y segunda derivada gaussiano
    # con el sigma, si no con el tamaño de mascara
    if sigma != None:
        mascara_gaussiana = kernel_gaussiano_1d(sigma=sigma)
        mascara_seg_gaussiana = kernel_gaussiano_1d(sigma=sigma, func=segunda_derivada_f_gaussiana)

    elif tam_mascara != None:
        mascara_gaussiana = kernel_gaussiano_1d(tam_mascara=tam_mascara)
        mascara_seg_gaussiana = kernel_gaussiano_1d(func=segunda_derivada_f_gaussiana, tam_mascara=tam_mascara)
    else:
        print("ERROR: Tienes que introducir un sigma o un t. mascara")

    continuar = sigma != None or tam_mascara != None

    imagen_laplaciana= []

    # aplicamos las convoluciones a la imagen, y sumamos las dos imagenes resultantes
    if continuar:
        # podemos utilizar el mismo kernel para obtener dxx y dyy porque la
        # derivada de la gaussiana es equivalente derivar por x que por y
        dxx = aplicar_convolucion (imagen, mascara_seg_gaussiana, mascara_gaussiana )
        dyy = aplicar_convolucion(imagen, mascara_gaussiana, mascara_seg_gaussiana)

        # la suma la multiplicamos por sigma al cuadrado para normalizar
        imagen_laplaciana = sigma**2 * (np.array(dxx) + np.array(dyy))

    return normaliza_imagen(imagen_laplaciana)



# probamos las mascaras
cat_s1 = mascara_laplaciana(cat, sigma=1)
cat_s3 = mascara_laplaciana(cat, sigma=3)

titulos = ["Original", "Máscara laplaciana sigma=1", "Máscara laplaciana sigma=3"]
mostrar_imagenes([cat, cat_s1, cat_s3], titulos)


input("\n---------- Pulsa una tecla para continuar ----------\n")

"""
Ejercicio 2

Apartado A
"""

def apilar_piramide(piramide):
    """
    Funcion para dada una piramide (secuencia de imagenes) unirla en una única imagen
    para mostrarla con los tamaños reales
    """
    # la anchura sera la anchura del nivel 1 (el 0 es la imagen orignial)
    anchura_primera_gaussiana = piramide[1].shape[1]

    imagen_final = piramide[1]

    # apilamos desde el nivel 1 hasta el N
    for i in range(2, len(piramide)):
        ajustada = np.ones((piramide[i].shape[0], anchura_primera_gaussiana))
        ajustada[:piramide[i].shape[0], :piramide[i].shape[1]] = piramide[i]

        imagen_final = np.vstack((imagen_final, ajustada))

    # añadimos por la izquierda la imagen original, la base de la piramide
    # teniendo en cuenta si es mas grande la imagen original o la union de los
    # niveles a la hora de unirlas
    if piramide[0].shape[0] > imagen_final.shape[0]:
        ajustada = np.ones((piramide[0].shape[0], imagen_final.shape[1]))
        ajustada[:imagen_final.shape[0], :imagen_final.shape[1]] = imagen_final
        imagen_final = ajustada

    elif piramide[0].shape[0] < imagen_final.shape[0]:
        ajustada = np.ones((imagen_final.shape[0], piramide[0].shape[1]))
        ajustada[:piramide[0].shape[0], :piramide[0].shape[1]] = piramide[0]
        piramide[0] = ajustada



    imagen_final = np.hstack((piramide[0], imagen_final))

    return imagen_final

def piramide_gaussiana(imagen, niveles=4, sigma_g=1, tipo_borde=cv.BORDER_REPLICATE):
    """
    Calcular la piramide gaussiana
    """

    solucion = []
    # ponemos la base de la piramide, el nivel 0 es la original
    solucion.append(imagen)

    # calculamos el kernel gaussiano a utilizar
    kernel = kernel_gaussiano_1d(sigma=sigma_g)


    # para cada nivel
    for i in range(niveles):
        # cogemos la ultima imagen calculada, y le aplicamos el alisamiento con
        # el kernel gaussiano
        imagen_con_blur = aplicar_convolucion(solucion[-1], kernel, kernel, tipo_borde)

        # cogemos los indices de las filas y columnas de dos en dos, eliminando
        # la mitad de las filas y de las columnas
        imagen_con_blur = imagen_con_blur[::2, ::2]

        # añadimos el nivel a la solucion
        solucion.append(imagen_con_blur)

    return solucion

def piramide_gaussiana_cv(imagen, niveles=4, tipo_borde=cv.BORDER_REPLICATE):
    """
    Funcion para calcular la piramide gaussiana con opencv. La usaremos para
    comparar soluciones
    """
    solucion = [imagen]

    # para cada nivel, añadimos lo devuelto por pyrDown con el nivel anterior
    for i in range(niveles):
        solucion.append(cv.pyrDown(solucion[-1], borderType=tipo_borde) )

    return solucion

# probamos y comparamos
piramide = piramide_gaussiana(einstein)

final = apilar_piramide(piramide)


piramide_cv = piramide_gaussiana_cv(einstein)

final_cv = apilar_piramide(piramide_cv)

mostrar_imagenes([final, final_cv], ["Implementación propia", "Utilizando pyrDown"])

input("\n---------- Pulsa una tecla para continuar ----------\n")

"""
Ejercicio 2

Apartado B

"""

def piramide_laplaciana(imagen, niveles=4, tipo_borde=cv.BORDER_REPLICATE):
    """
    Funcion para obtener la piramide laplaciana
    """

    # calculamos la piramide gaussiana
    p_gaussiana = piramide_gaussiana(imagen, niveles, tipo_borde)

    solucion = []

    # para cada nivel
    for i in range(niveles):
        # cogemos la gaussiana del siguiente nivel
        img_gaussiana = p_gaussiana[i + 1]
        forma = (p_gaussiana[i].shape[1], p_gaussiana[i].shape[0])
        # la reescalamos usando cv.resize
        img_gaussiana = cv.resize(src=img_gaussiana, dsize=forma)

        # la restamos con la gaussiana del nivel anterior
        laplaciana = p_gaussiana[i] - img_gaussiana

        # y normalizamos la imagen
        laplaciana = normaliza_imagen(laplaciana)

        solucion.append(laplaciana)

    # por ultimo, añadimos el nivel más pequeño de la gaussiana
    solucion.append(p_gaussiana[-1])

    return solucion

def piramide_laplaciana_cv(imagen, niveles=4, tipo_borde=cv.BORDER_REPLICATE):
    """
    Funcion para obtener la piramide laplaciana utilizando opencv. Usaremos
    esta funcion para comparar resultados
    """
    # calculamos la piramide gaussiana utilizando opencv
    p_gaussiana = piramide_gaussiana_cv(imagen, niveles, tipo_borde)

    # la piramide va a estar invertida, luego le daremos la vuelta
    solucion = [p_gaussiana[-1]]

    for i in range(niveles):
        # usamos pyrUp para reescalar el nivel anterior
        forma = (p_gaussiana[-(i+2)].shape[1], p_gaussiana[-(i+2)].shape[0])
        reescalada = cv.pyrUp(p_gaussiana[-(i+1)], dstsize=forma)

        # la restamos con el nivel anterior
        laplaciana = p_gaussiana[-(i+2)] - reescalada

        # la normalizamos y la añadimos
        laplaciana = normaliza_imagen(laplaciana)

        solucion.append(laplaciana)

    # como la hemos calculado al reves, la invertimos
    solucion = solucion[::-1]

    return solucion


# probamos y comprobamos

piramide = piramide_laplaciana(bicycle)

final = apilar_piramide(piramide)

piramide_cv = piramide_laplaciana_cv(bicycle)

final_cv = apilar_piramide(piramide_cv)

mostrar_imagenes([final, final_cv], ["Implementación propia", "Utilizando pyrUp"])


input("\n---------- Pulsa una tecla para continuar ----------\n")


"""
Ejercicio 3

"""


def crear_imagen_hibrida(imagen_f_bajas, imagen_f_altas, sigma_img_f_bajas, sigma_img_f_altas, t_borde=cv.BORDER_REPLICATE):

    kernel_f_bajas = kernel_gaussiano_1d(sigma=sigma_img_f_bajas)
    kernel_f_altas = kernel_gaussiano_1d(sigma=sigma_img_f_altas)

    imagen_f_bajas = aplicar_convolucion(imagen_f_bajas, kernel_f_bajas, kernel_f_bajas, t_borde)
    imagen_f_altas_sin_f_bajas = aplicar_convolucion(imagen_f_altas, kernel_f_altas, kernel_f_altas, t_borde)

    imagen_f_altas = imagen_f_altas - imagen_f_altas_sin_f_bajas


    imagen_hibrida = imagen_f_bajas + imagen_f_altas

    imagen_hibrida = normaliza_imagen(imagen_hibrida)
    imagen_f_bajas = normaliza_imagen(imagen_f_bajas)
    imagen_f_altas = normaliza_imagen(imagen_f_altas)

    return imagen_hibrida, imagen_f_bajas, imagen_f_altas




#imagen_hibrida, img_f_bajas, img_f_altas = crear_imagen_hibrida(bird, plane, 3, 2)
imagen_hibrida, img_f_bajas, img_f_altas = crear_imagen_hibrida(einstein, marilyn, 3, 1.5)


mostrar_imagenes([imagen_hibrida, img_f_bajas, img_f_altas])







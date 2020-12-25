#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

def leeimagen(fichero, flag_color):

    imagen = cv.imread(fichero, flag_color)

    imagen = imagen.astype(np.float32)

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
        forma = (piramide[i].shape[0], anchura_primera_gaussiana)

        if piramide[0].ndim == 3:
            forma = (piramide[i].shape[0], anchura_primera_gaussiana, 3)

        ajustada = np.ones(forma)

        ajustada[:piramide[i].shape[0], :piramide[i].shape[1]] = piramide[i]

        imagen_final = np.vstack((imagen_final, ajustada))

    # añadimos por la izquierda la imagen original, la base de la piramide
    # teniendo en cuenta si es mas grande la imagen original o la union de los
    # niveles a la hora de unirlas
    if piramide[0].shape[0] > imagen_final.shape[0]:
        forma = (piramide[0].shape[0], imagen_final.shape[1])

        if piramide[0].ndim == 3:
            forma = (piramide[0].shape[0], imagen_final.shape[1], 3)

        ajustada = np.ones(forma)
        ajustada[:imagen_final.shape[0], :imagen_final.shape[1]] = imagen_final
        imagen_final = ajustada

    elif piramide[0].shape[0] < imagen_final.shape[0]:
        forma = (imagen_final.shape[0], piramide[0].shape[1])

        if piramide[0].ndim == 3:
            forma = (imagen_final.shape[0], piramide[0].shape[1], 3)

        ajustada = np.ones(forma)
        ajustada[:piramide[0].shape[0], :piramide[0].shape[1]] = piramide[0]
        piramide[0] = ajustada



    imagen_final = np.hstack((piramide[0], imagen_final))

    return imagen_final

def piramide_gaussiana_cv(imagen, niveles=3, tipo_borde=cv.BORDER_REPLICATE):
    """
    Funcion para calcular la piramide gaussiana con opencv.
    """
    solucion = [imagen]

    # para cada nivel, añadimos lo devuelto por pyrDown con el nivel anterior
    for i in range(niveles):
        solucion.append(cv.pyrDown(solucion[-1], borderType=tipo_borde) )

    return solucion



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



def aplicar_convolucion(imagen, k_x, k_y):

    k_x_invertido = np.flip(k_x)
    k_y_invertido = np.flip(k_y)

    img_conv_c = cv.filter2D(imagen, -1, k_x_invertido)
    img_conv_final = cv.filter2D(img_conv_c, -1, k_y_invertido)

    return img_conv_final


def supresion_no_maximos(imagen, tam_bloque):

    # el resultado será una imagen en blanco
    resultado = np.zeros(imagen.shape);

    # calculamos el rango a mirar, es la mitad ya que iremos de [-rango, rango]
    rango = tam_bloque // 2

    # para los indices de laimagen
    for i, j in np.ndindex(imagen.shape):

        # tenemos en cuenta los bordes
        tope_inf_x = max(i - rango, 0)
        tope_sup_x = i + rango + 1

        tope_inf_y = max( j - rango, 0 )
        tope_sup_y = j + rango + 1

        # calculamos la ventana del rango actual
        ventana = imagen[tope_inf_x:tope_sup_x, tope_inf_y:tope_sup_y]

        # si el actual es un maximo, lo guardamos en la imagen
        if np.max(ventana) == imagen[i, j]:
            resultado[i, j] = imagen[i, j]

    return resultado

def piramide_derivada_gaussiana(imagen, k_size, tam_piramide, sigma):

    kernel = kernel_gaussiano_1d(sigma)
    img_alisada = aplicar_convolucion(imagen, kernel, kernel)

    mascara_x_kx, mascara_y_kx = cv.getDerivKernels(1, 0, k_size, normalize = True)
    mascara_x_ky, mascara_y_ky = cv.getDerivKernels(0, 1, k_size, normalize = True)

    img_dx = aplicar_convolucion(img_alisada, mascara_x_kx, mascara_y_kx)
    img_dy = aplicar_convolucion(img_alisada, mascara_x_ky, mascara_y_ky)

    piramide_dx = [img_dx]
    piramide_dy = [img_dy]

    for i in range(1, tam_piramide):
        piramide_dx.append(cv.pyrDown(piramide_dx[i-1]))
        piramide_dy.append(cv.pyrDown(piramide_dy[i-1]))

    return piramide_dx, piramide_dy


def puntos_interes(imagen, tam_bloque, k_size):

    val_eigen = cv.cornerEigenValsAndVecs(imagen, tam_bloque, k_size)

    # nos quedamos con los valores singulares
    val_eigen = val_eigen[:, :, :2]

    producto = np.prod(val_eigen, axis = 2)
    suma = np.sum(val_eigen, axis = 2)

    # hacemos la division de los productos y la suma, y la salida será una matriz de ceros a excepción de donde la suma sera 0, para no dividir por 0
    puntos_interes = np.divide(producto, suma, out = np.zeros(imagen.shape), where = suma != 0.0)

    return puntos_interes


def orientacion_gradiente(grad_x, grad_y):

    vectores_u = np.concatenate([ grad_x.reshape(-1, 1), grad_y.reshape(-1,1) ], axis = 1)
    normas_vectores_u = np.linalg.norm(vectores_u, axis = 1)

    sin_cos_vectores_u = vectores_u / normas_vectores_u.reshape(-1, 1)
    cosenos = sin_cos_vectores_u[:, 0]
    senos = sin_cos_vectores_u[:, 1]

    orientacion = np.divide(senos, cosenos, out = np.zeros(senos.shape), where = cosenos != 0.0)

    radianes = np.arctan(orientacion)

    grados = np.degrees(radianes)

    grados[cosenos < 0.0] += 180
    grados[grados < 0.0] += 360

    return grados

def puntos_harris(imagen, tam_bloque, tam_ventana, num_escalas, sigma_p_gauss, umbral_harris, ksize):

    piramide_gauss = piramide_gaussiana_cv(imagen, num_escalas)

    sigma = 4.5

    piramide_derivada_x, piramide_derivada_y = piramide_derivada_gaussiana(imagen, ksize, num_escalas, sigma)

    puntos_harris = []
    puntos_harris_corregidos = []

    for i in range(num_escalas):

        p_interes = puntos_interes(piramide_gauss[i], tam_bloque, ksize)


        # ponemos a 0 los puntos que no cumplen con el umbral
        p_interes[p_interes < umbral_harris] = 0.0

        p_interes = supresion_no_maximos(p_interes, tam_ventana)

        puntos_a_usar = np.where(p_interes > 0.0)

        derivadas_no_eliminados_x = piramide_derivada_x[i][puntos_a_usar]
        derivadas_no_eliminados_y = piramide_derivada_y[i][puntos_a_usar]

        escala_puntos = (i + 1) * tam_bloque

        orientacion_puntos = orientacion_gradiente( derivadas_no_eliminados_x, derivadas_no_eliminados_y )

        puntos_escala = []

        for y, x, o in zip(*puntos_a_usar, orientacion_puntos):
            puntos_escala.append(cv.KeyPoint(float(x)*2**i, float(y)*2**i, escala_puntos, o))


        puntos_x = puntos_a_usar[0].reshape(-1, 1)
        puntos_y = puntos_a_usar[1].reshape(-1, 1)

        puntos = np.concatenate([puntos_x, puntos_y], axis = 1)

        # paramos con 15 iteraciones o con epsilon < 0.01
        criterio = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 0.01)

        puntos = cv.cornerSubPix(piramide_gauss[i], puntos.astype(np.float32), (3,3), (-1, -1), criterio)

        puntos = np.round(puntos)
        puntos = np.flip(puntos, axis = 1)
        puntos *= 2**i


        puntos_harris.append(puntos_escala)
        puntos_harris_corregidos.append(puntos)

    return puntos_harris, puntos_harris_corregidos


def normaliza_imagen_255( imagen ):

    if imagen.ndim == 2:
        min_val = np.min(imagen)
        max_val = np.max(imagen)
    else:
        min_val = np.min(imagen, axis=(0, 1))
        max_val = np.max(imagen, axis=(0, 1))


    # Normalizar la imagen al rango [0, 1]
    norm = (imagen - min_val) / (max_val - min_val)

    # Multiplicar cada pixel por 255
    norm = norm * 255

    # Redondear los valores y convertirlos a uint8
    resultado = np.round(norm).astype(np.uint8)

    return resultado



def dibujar_puntos_harris( imagen, puntos ):

    todos_puntos = []

    # juntamos todos los puntos de las distintas escalas
    for escala in puntos:
        for punto in escala:
            todos_puntos.append(punto)


    imagen = normaliza_imagen_255(imagen)

    img_con_puntos = np.empty(imagen.shape)

    img_con_puntos = cv.drawKeypoints(imagen, todos_puntos, img_con_puntos, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_con_puntos


"""
Apartado 1
"""

yosemite_1_bn = leeimagen("imagenes/Yosemite1.jpg", 0)
yosemite_1_color = leeimagen("imagenes/Yosemite1.jpg", 1)

yosemite_2_bn = leeimagen("imagenes/Yosemite2.jpg", 0)
yosemite_2_color = leeimagen("imagenes/Yosemite2.jpg", 1)

puntos, puntos_corregidos = puntos_harris(yosemite_1_bn, tam_bloque = 5, tam_ventana = 3, num_escalas = 3, sigma_p_gauss = 4.5, umbral_harris = 10.0, ksize = 3)



imagen_con_puntos = dibujar_puntos_harris(yosemite_1_color, puntos)

mostrar_imagen(imagen_con_puntos)

puntos_u90, puntos_corregidos_u90 = puntos_harris(yosemite_1_bn, tam_bloque = 5, tam_ventana = 3, num_escalas = 3, sigma_p_gauss = 4.5, umbral_harris = 90.0, ksize = 3)



imagen_con_puntos = dibujar_puntos_harris(yosemite_1_color, puntos_u90)

mostrar_imagen(imagen_con_puntos)




"""
Apartado 2
"""

def puntos_descriptores_AKAZE(imagen, umbral):

    akaze = cv.AKAZE_create(threshold = umbral)

    puntos_clave, descriptores = akaze.detectAndCompute(imagen, None)

    return puntos_clave, descriptores


def coincidencias_descriptores_fuerza_bruta(descriptores1, descriptores2):

    emparejador = cv.BFMatcher_create(crossCheck = True)

    coincidencias = emparejador.match(descriptores1, descriptores2)

    return coincidencias


def coincidencias_descriptores_2nn(descriptores1, descriptores2):

    emparejador = cv.BFMatcher_create()

    coincidencias = emparejador.knnMatch(descriptores1, descriptores2, k = 2)

    return coincidencias



def coincidencias_descriptores_lowe_average_2nn(descriptores1, descriptores2):

    coincidencias = coincidencias_descriptores_2nn(descriptores1, descriptores2)

    coincidencias_lowe = []

    # usamos el criterio de lowe para el mejor coincidencia
    for coincidencia_x, coincidencia_y in coincidencias:
        if coincidencia_x.distance < 0.8 * coincidencia_y.distance:
            coincidencias_lowe.append(coincidencia_x)

    return coincidencias_lowe



def dibujar_coincidencias(imagen1, imagen2, puntos_clave1, puntos_clave2, coincidencias, a_mostrar = 100):


    imagen1 = normaliza_imagen_255(imagen1)
    imagen2 = normaliza_imagen_255(imagen2)

    # juntamos las dos imagenes
    imagen_resultado = np.concatenate([imagen1, imagen2], axis=1)

    coincidencias_aleatorias = np.random.choice(coincidencias, a_mostrar, replace = False)


    imagen_resultado = cv.drawMatches(imagen1, puntos_clave1, imagen2, puntos_clave2, coincidencias_aleatorias, imagen_resultado, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    return imagen_resultado




puntos_yosemite1, desciptores_yosemite1 = puntos_descriptores_AKAZE(yosemite_1_bn, 0.1)
puntos_yosemite2, desciptores_yosemite2 = puntos_descriptores_AKAZE(yosemite_2_bn, 0.1)


coincidencias_fuerza_bruta = coincidencias_descriptores_fuerza_bruta(desciptores_yosemite1, desciptores_yosemite2)

coincidencias_lowe = coincidencias_descriptores_lowe_average_2nn(desciptores_yosemite1, desciptores_yosemite2)

resultado_fuerza_bruta = dibujar_coincidencias(yosemite_1_color, yosemite_2_color, puntos_yosemite1, puntos_yosemite2, coincidencias_fuerza_bruta)


resultado_lowe = dibujar_coincidencias(yosemite_1_color, yosemite_2_color, puntos_yosemite1, puntos_yosemite2, coincidencias_lowe)

mostrar_imagen(resultado_fuerza_bruta)

mostrar_imagen(resultado_lowe)




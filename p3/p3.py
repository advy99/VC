#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

"""
Funciones auxiliares (de otras prácticas, principalmente p1)
"""


def leeimagen(fichero, flag_color):

    """
    Leer una imagen de un fichero (pasado como string), tambien se pasa un flag para leer a color o en B/N
    """

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
    """
    Aplicar una convolucion a una imagen usando opencv
    """

    # invertimos los kernels, para que sea una convolucion y no una correlacion
    k_x_invertido = np.flip(k_x)
    k_y_invertido = np.flip(k_y)

    # aplicamos el primero
    img_conv_c = cv.filter2D(imagen, -1, k_x_invertido)

    # a la solucion aplicamos el segundo
    img_conv_final = cv.filter2D(img_conv_c, -1, k_y_invertido)

    return img_conv_final




"""
Apartado 1
"""



def supresion_no_maximos(imagen, tam_bloque):

    """
    Funcion para realizar la supresión de no maximos de una imagen
    """

    # el resultado será una imagen en blanco
    resultado = np.zeros(imagen.shape);

    # calculamos el rango a mirar, es la mitad ya que iremos de [-rango, rango]
    rango = tam_bloque // 2

    # para los indices de la imagen
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

    """
    Pirámide con las derivadas de la gaussiana de la imagen. La usaremos
    para conocer los gradientes de una imagen a distintas escalas
    """

    # alisamos la imagen
    kernel = kernel_gaussiano_1d(sigma)
    img_alisada = aplicar_convolucion(imagen, kernel, kernel)

    # obtenemos los kernels para cada eje
    mascara_x_kx, mascara_y_kx = cv.getDerivKernels(1, 0, k_size, normalize = True)
    mascara_x_ky, mascara_y_ky = cv.getDerivKernels(0, 1, k_size, normalize = True)

    # aplicamos las convoluciones
    img_dx = aplicar_convolucion(img_alisada, mascara_x_kx, mascara_y_kx)
    img_dy = aplicar_convolucion(img_alisada, mascara_x_ky, mascara_y_ky)


    # construimos la piramide con el primer nivel con las derivadas
    piramide_dx = [img_dx]
    piramide_dy = [img_dy]

    for i in range(1, tam_piramide):
        piramide_dx.append(cv.pyrDown(piramide_dx[i-1]))
        piramide_dy.append(cv.pyrDown(piramide_dy[i-1]))

    return piramide_dx, piramide_dy


def puntos_interes(imagen, tam_bloque, k_size):

    """
    Funcion para obtener los puntos de interes de una imagen
    """

    # obtenemos los valores con opencv
    val_eigen = cv.cornerEigenValsAndVecs(imagen, tam_bloque, k_size)

    # nos quedamos con los valores singulares
    val_eigen = val_eigen[:, :, :2]

    # calculamos el producto y la suma de los valores singulares
    producto = np.prod(val_eigen, axis = 2)
    suma = np.sum(val_eigen, axis = 2)

    # hacemos la division de los productos y la suma, y la salida será una matriz de ceros a excepción de donde la suma sera 0, para no dividir por 0
    puntos_interes = np.divide(producto, suma, out = np.zeros(imagen.shape), where = suma != 0.0)

    return puntos_interes


def orientacion_gradiente(grad_x, grad_y):

    """
    Funcion para calcular la orientación de un gradiente dado
    """

    # guardamos los vectores dados en forma de un vector de una fila
    vectores_u = np.concatenate([ grad_x.reshape(-1, 1), grad_y.reshape(-1,1) ], axis = 1)

    # calculamos las normales del vector
    normas_vectores_u = np.linalg.norm(vectores_u, axis = 1)

    # calculamos los senos y cosenos dividiendo el vector por su norma
    sin_cos_vectores_u = vectores_u / normas_vectores_u.reshape(-1, 1)
    cosenos = sin_cos_vectores_u[:, 0]
    senos = sin_cos_vectores_u[:, 1]

    # la orientacion es el seno entre el coseno, y tenemos cuidado de no dividir por 0
    orientacion = np.divide(senos, cosenos, out = np.zeros(senos.shape), where = cosenos != 0.0)

    # le aplicamos la arctan para pasarlo a radianes
    radianes = np.arctan(orientacion)

    # y lo pasamos a grados
    grados = np.degrees(radianes)

    # nos aseguramos que estan en el rango 0 - 360
    grados[cosenos < 0.0] += 180
    grados[grados < 0.0] += 360

    return grados

def puntos_harris(imagen, tam_bloque, tam_ventana, num_escalas, sigma_p_gauss, umbral_harris, ksize):

    """
    Funcion para calcular los puntos harris de una imagen
    """

    # calculamos la piramide gaussiana
    piramide_gauss = piramide_gaussiana_cv(imagen, num_escalas)


    # calculamos la piramide de las derivadas
    piramide_derivada_x, piramide_derivada_y = piramide_derivada_gaussiana(imagen, ksize, num_escalas, sigma_p_gauss)

    puntos_harris = []
    puntos_harris_corregidos = []

    # para el numero de escalas que nos piden
    for i in range(num_escalas):

        # calculamos los puntos de interes de ese nivel de la escala
        p_interes = puntos_interes(piramide_gauss[i], tam_bloque, ksize)


        # ponemos a 0 los puntos que no cumplen con el umbral
        p_interes[p_interes < umbral_harris] = 0.0

        # aplicamos la supresion de no maximos a los puntos de interes
        p_interes = supresion_no_maximos(p_interes, tam_ventana)

        # nos quedamos con los positivos
        puntos_a_usar = np.where(p_interes > 0.0)

        # guardamos las derivadas de los puntos a usar en la escala actual
        derivadas_no_eliminados_x = piramide_derivada_x[i][puntos_a_usar]
        derivadas_no_eliminados_y = piramide_derivada_y[i][puntos_a_usar]

        # estimacion de la escala de los puntos, como nos dice en el guion
        escala_puntos = 2**(i) * tam_bloque

        # calculamos la orientacion de los puntos actuales
        orientacion_puntos = orientacion_gradiente( derivadas_no_eliminados_x, derivadas_no_eliminados_y )

        puntos_escala = []

        # calculamos todos los puntos de una escala
        for y, x, o in zip(*puntos_a_usar, orientacion_puntos):
            puntos_escala.append(cv.KeyPoint(float(x)*2**i, float(y)*2**i, escala_puntos, o))


        # separamos la x e y de los puntos a usar
        puntos_x = puntos_a_usar[0].reshape(-1, 1)
        puntos_y = puntos_a_usar[1].reshape(-1, 1)

        # los ponemos en un vector fila
        puntos = np.concatenate([puntos_x, puntos_y], axis = 1)

        # paramos con 15 iteraciones o con epsilon < 0.01
        criterio = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 0.01)

        # aplicamos cornerSubPix  para obtener los puntos corregidos
        puntos = cv.cornerSubPix(piramide_gauss[i], puntos.astype(np.float32), (3,3), (-1, -1), criterio)

        # aplicamos la correccion a los puntos
        puntos = np.round(puntos)
        puntos = np.flip(puntos, axis = 1)
        puntos *= 2**i


        # guardamos los puntos y los puntos corregidos de la escala actual
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
    i = 0

    print("\n")

    # juntamos todos los puntos de las distintas escalas
    for escala in puntos:
        print("La escala ", i, " tiene ", len(escala), " puntos")
        i += 1

        for punto in escala:
            todos_puntos.append(punto)


    # normalizamos la imagen
    imagen = normaliza_imagen_255(imagen)

    img_con_puntos = np.empty(imagen.shape)

    # utilizamos drawKeyPoints para dibujar los puntos
    # utilizamos el flag DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS para diferenciar las distintas escalas
    img_con_puntos = cv.drawKeypoints(imagen, todos_puntos, img_con_puntos, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_con_puntos


np.random.seed(123)

print("\n Apartado 1 \n")

# probamos el codigo
yosemite_1_bn = leeimagen("imagenes/Yosemite1.jpg", 0)
yosemite_1_color = leeimagen("imagenes/Yosemite1.jpg", 1)

yosemite_2_bn = leeimagen("imagenes/Yosemite2.jpg", 0)
yosemite_2_color = leeimagen("imagenes/Yosemite2.jpg", 1)

# sacamos los puntos con la imagen en B/N y los pintamos en la imagen a color
# ya que no podemos obtener los puntos de la imagen a color
puntos, puntos_corregidos = puntos_harris(yosemite_1_bn, tam_bloque = 5, tam_ventana = 7, num_escalas = 3, sigma_p_gauss = 4.5, umbral_harris = 10.0, ksize = 3)

imagen_con_puntos = dibujar_puntos_harris(yosemite_1_color, puntos)

mostrar_imagen(imagen_con_puntos)

input("---------- Pulsa una tecla para continuar ----------")


# probamos un umbral más grande
puntos, puntos_corregidos = puntos_harris(yosemite_1_bn, tam_bloque = 5, tam_ventana = 7, num_escalas = 3, sigma_p_gauss = 4.5, umbral_harris = 300.0, ksize = 3)

imagen_con_puntos = dibujar_puntos_harris(yosemite_1_color, puntos)

mostrar_imagen(imagen_con_puntos)

input("---------- Pulsa una tecla para continuar ----------")

puntos, puntos_corregidos = puntos_harris(yosemite_2_bn, tam_bloque = 5, tam_ventana = 7, num_escalas = 3, sigma_p_gauss = 4.5, umbral_harris = 10.0, ksize = 3)

imagen_con_puntos = dibujar_puntos_harris(yosemite_2_color, puntos)

mostrar_imagen(imagen_con_puntos)

input("---------- Pulsa una tecla para continuar ----------")

# comparacion de puntos y puntos corregidos

coordenadas_puntos = []
coordenadas_puntos_corregidos = []

for punto in puntos:
    for k in punto:
        coordenadas_puntos.append(list(k.pt))


for punto in puntos_corregidos:
    for k in punto:
        coordenadas_puntos_corregidos.append(k)

coordenadas_puntos = np.array(coordenadas_puntos)
coordenadas_puntos_corregidos = np.array(coordenadas_puntos_corregidos)

# nos quedamos con los puntos distintos
distinto_x = coordenadas_puntos[:, 0] != coordenadas_puntos_corregidos[:, 0]
distinto_y = coordenadas_puntos[:, 1] != coordenadas_puntos_corregidos[:, 1]

distintos = distinto_x * distinto_y

puntos_distintos = np.where(distintos == True)

# copiamos la imagen original
comparar_puntos = np.copy(yosemite_1_color)

comparar_puntos = normaliza_imagen_255(comparar_puntos)

# para tres puntos distintos
for aleatorio in np.random.choice(puntos_distintos[0], 3, replace = False):

    original = coordenadas_puntos[aleatorio]
    corregido = coordenadas_puntos_corregidos[aleatorio]

    original = original.astype(np.uint8)
    corregido = corregido.astype(np.uint8)

    # dibujamos el punto original y corregido como circulos en la imagen
    comparar_puntos = cv.circle(comparar_puntos, tuple(original), 2, (255, 0, 0))
    comparar_puntos = cv.circle(comparar_puntos, tuple(corregido), 2, (0, 0, 255))


    px = original[0]
    py = original[1]

    # calculamos el rango de ventana 9x9 que queremos ver
    rango = np.array([px - 4, px + 5, py - 4, py + 5])
    rango = rango.astype(int)

    rango[rango < 0] = 0

    # y lo mostramos
    ventana = comparar_puntos[rango[2]:rango[3], rango[0]:rango[1]]

    mostrar_imagen( ventana )

    input("---------- Pulsa una tecla para continuar ----------")


"""
Apartado 2
"""

def puntos_descriptores_AKAZE(imagen, umbral):
    """
    Funcion para obtener los puntos clave y descriptores usando AKAZE
    """

    akaze = cv.AKAZE_create(threshold = umbral)

    puntos_clave, descriptores = akaze.detectAndCompute(imagen, None)

    return puntos_clave, descriptores


def coincidencias_descriptores_fuerza_bruta(descriptores1, descriptores2):
    """
    Funcion para obtener las correspondencias entre dos descriptores usando
    fuerza bruta
    """
    emparejador = cv.BFMatcher_create(crossCheck = True)

    coincidencias = emparejador.match(descriptores1, descriptores2)

    return coincidencias


def coincidencias_descriptores_2nn(descriptores1, descriptores2):
    """
    Funcion para obtener las correspondencias entre dos descriptores usando
    un 2NN
    """
    emparejador = cv.BFMatcher_create()

    coincidencias = emparejador.knnMatch(descriptores1, descriptores2, k = 2)

    return coincidencias



def coincidencias_descriptores_lowe_average_2nn(descriptores1, descriptores2):
    """
    Funcion para obtener las correspondencias entre dos descriptores usando
    2NN y aplicando el criterio de Lowe
    """
    coincidencias = coincidencias_descriptores_2nn(descriptores1, descriptores2)

    coincidencias_lowe = []

    # usamos el criterio de lowe para el mejor coincidencia
    for coincidencia_x, coincidencia_y in coincidencias:
        if coincidencia_x.distance < 0.8 * coincidencia_y.distance:
            coincidencias_lowe.append(coincidencia_x)

    return coincidencias_lowe



def dibujar_coincidencias(imagen1, imagen2, puntos_clave1, puntos_clave2, coincidencias, a_mostrar = 100):
    """
    Funcion para obtener una imagen con las coincidencias entre dos imagenes,
    dadas las imagenes, sus puntos clave, y las coincidencias. Como parametro opcional, se nos puede dar el numero de coincidencias a dibujar, por defecto 100
    """


    imagen1 = normaliza_imagen_255(imagen1)
    imagen2 = normaliza_imagen_255(imagen2)

    # juntamos las dos imagenes
    imagen_resultado = np.concatenate([imagen1, imagen2], axis=1)

    # escogemos a_mostrar aleatorias
    coincidencias_aleatorias = np.random.choice(coincidencias, a_mostrar, replace = False)

    # las dibujamos usando drawMatches
    imagen_resultado = cv.drawMatches(imagen1, puntos_clave1, imagen2, puntos_clave2, coincidencias_aleatorias, imagen_resultado, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    return imagen_resultado


print("\n Apartado 2 \n")

# probamos la funcion
puntos_yosemite1, descriptores_yosemite1 = puntos_descriptores_AKAZE(yosemite_1_bn, 0.01)
puntos_yosemite2, descriptores_yosemite2 = puntos_descriptores_AKAZE(yosemite_2_bn, 0.01)


coincidencias_fuerza_bruta = coincidencias_descriptores_fuerza_bruta(descriptores_yosemite1, descriptores_yosemite2)

coincidencias_lowe = coincidencias_descriptores_lowe_average_2nn(descriptores_yosemite1, descriptores_yosemite2)

resultado_fuerza_bruta = dibujar_coincidencias(yosemite_1_color, yosemite_2_color, puntos_yosemite1, puntos_yosemite2, coincidencias_fuerza_bruta)


resultado_lowe = dibujar_coincidencias(yosemite_1_color, yosemite_2_color, puntos_yosemite1, puntos_yosemite2, coincidencias_lowe)

mostrar_imagen(resultado_fuerza_bruta)
input("---------- Pulsa una tecla para continuar ----------")

mostrar_imagen(resultado_lowe)
input("---------- Pulsa una tecla para continuar ----------")






"""
Apartado 3 y 4
"""




def panorama_imagenes(imagenes):

    """
    Funcion para crear un panorama entre N imagenes
    """

    # se supone que estan ordenadas de derecha a izquierda
    # centro sera la posicion donde esta en imagenes la imagen central
    centro = len(imagenes) // 2

    # calculamos el tamaño para el resultado
    # aqui nos ponemos en el peor de los casos y creamos el resultado
    # más grande de lo que deberíamos, luego lo ajustamos
    tam_resul_x = 0
    tam_resul_y = 0

    for imagen in imagenes:
        tam_resul_x += imagen.shape[1]
        tam_resul_y += imagen.shape[0]

    desp_homo_x = tam_resul_x // 2
    desp_homo_y = tam_resul_y // 2

    resultado = np.zeros( ( tam_resul_x, tam_resul_y ) , dtype = np.uint8 )
    resultado = cv.cvtColor(resultado, cv.COLOR_BGR2RGB)

    imagen_central = normaliza_imagen_255(imagenes[centro])

    # creamos la homografia inicial. Como vamos a comenzar con la imagen central
    # el desplazamiento es la mitad del cuadro de resultado generado, para
    # ponerla en el centro
    homografia = np.array( [[1, 0, desp_homo_x],
                            [0, 1, desp_homo_y],
                            [0, 0, 1]], dtype = np.float64 )

    # ponemos la imagen central
    resultado = cv.warpPerspective(imagen_central, homografia, (tam_resul_x, tam_resul_y), dst=resultado, borderMode = cv.BORDER_TRANSPARENT)


    # volvemos a la homografia inicial, ya que empezamos desde el inicio, pero
    # hacia la izquierda
    copia_homografia = np.copy(homografia)

    # parte izquierda del panorama
    for i in range(centro, 0, -1):
        destino = imagenes[i]
        fuente = imagenes[i - 1]

        # sacamos puntos de interes y descriptores
        puntos_interes_destino, descriptores_destino = puntos_descriptores_AKAZE(destino, 0.01)
        puntos_interes_fuente, descriptores_fuente = puntos_descriptores_AKAZE(fuente, 0.01)

        # sacamos coincidencias
        coincidencias = coincidencias_descriptores_lowe_average_2nn(descriptores_destino, descriptores_fuente)

        puntos_destino = []
        puntos_fuente = []

        # sacamos los puntos de las coincidencias
        for coincidencia in coincidencias:
            puntos_destino.append(puntos_interes_destino[coincidencia.queryIdx].pt)
            puntos_fuente.append(puntos_interes_fuente[coincidencia.trainIdx].pt)

        puntos_destino = np.array(puntos_destino, dtype = np.float32)
        puntos_fuente = np.array(puntos_fuente, dtype = np.float32)

        # al igual que antes, obtenemos la homografia
        homografia_cv, _ = cv.findHomography(puntos_fuente, puntos_destino, cv.RANSAC, 5)
        # la apilamos con las anteriores
        copia_homografia = np.dot(copia_homografia, homografia_cv)

        copia_fuente = normaliza_imagen_255(fuente)

        # y la aplicamos
        resultado = cv.warpPerspective(copia_fuente, copia_homografia, (tam_resul_x, tam_resul_y), dst=resultado, borderMode = cv.BORDER_TRANSPARENT)

    # guardamos el extremo utilizado por la homografia, para recortar la imagen.
    # en este caso no necesitamos aplicar un ajuste ya que la imagen se ha colocado
    # en la derecha de este punto, y nos interesa el extremo izquierdo
    ancho_min = copia_homografia[1][2]
    alto_min = copia_homografia[0][2]

    ancho_min = int(ancho_min)
    alto_min = int(alto_min)



    # hacemos una copia de la homografia
    copia_homografia = np.copy(homografia)

    # parte derecha del panorama
    for i in range ( centro, len(imagenes) - 1 ):

        destino = imagenes[i]
        fuente = imagenes[i + 1]

        # sacamos los puntos y descriptores de cada imagen
        puntos_interes_destino, descriptores_destino = puntos_descriptores_AKAZE(destino, 0.01)
        puntos_interes_fuente, descriptores_fuente = puntos_descriptores_AKAZE(fuente, 0.01)

        # las coincidencias con lowe 2NN
        coincidencias = coincidencias_descriptores_lowe_average_2nn(descriptores_destino, descriptores_fuente)

        puntos_destino = []
        puntos_fuente = []

        # sacamos los puntos de destino y fuente donde hay coincidencias
        for coincidencia in coincidencias:
            puntos_destino.append(puntos_interes_destino[coincidencia.queryIdx].pt)
            puntos_fuente.append(puntos_interes_fuente[coincidencia.trainIdx].pt)

        puntos_destino = np.array(puntos_destino, dtype = np.float32)
        puntos_fuente = np.array(puntos_fuente, dtype = np.float32)


        # obtenemos la homografia donde iría la imagen
        homografia_cv, _ = cv.findHomography(puntos_fuente, puntos_destino, cv.RANSAC, 5)
        # aplicamos una multiplicacion matricual para actualizar la homografia
        # que estamos usando, de cara a que se apilen las transformaciones
        # de forma que se vayan posicionando de forma correcta el resultado
        copia_homografia = np.dot(copia_homografia, homografia_cv)

        copia_fuente = normaliza_imagen_255(fuente)

        # por ultimo, añadimos la imagen al resultado con warpPerspective
        resultado = cv.warpPerspective(copia_fuente, copia_homografia, (tam_resul_x, tam_resul_y), dst=resultado, borderMode = cv.BORDER_TRANSPARENT)


    # cuando hacemos todas las de la parte derecha, sabemos hasta donde ha llegado
    # para recortar el resultado. Añadimos cierto margen, ya que al deformar las
    # imagenes y el propio tamaño de la imagen hace que necesitemos más espacio
    ancho_max = copia_homografia[1][2] + imagenes[len(imagenes) - 1].shape[0] * 1.4
    alto_max = copia_homografia[0][2] + imagenes[len(imagenes) - 1].shape[1] * 1.4

    # lo pasamos a entero
    ancho_max = int(ancho_max)
    alto_max = int(alto_max)

    # el resultado es el resultado original, pero recortando toda la zona negra sin utilizar
    resultado = resultado[ancho_min:ancho_max, alto_min:alto_max ]

    return resultado

def panorama_3_imagenes(imagen1, imagen2, imagen3):
    """
    Funcion para calcular el panorama de dos imagenes. Simplemente llamamos
    a la de N imagenes, pero con dos
    """
    return panorama_imagenes([imagen1, imagen2])

def panorama_2_imagenes(imagen1, imagen2):
    """
    Funcion para calcular el panorama de dos imagenes. Simplemente llamamos
    a la de N imagenes, pero con dos
    """
    return panorama_imagenes([imagen1, imagen2])


# leemos las imagenes de la etsiit
mosaico_etsiit = []

for i in range(2, 10):
    mosaico_etsiit.append("imagenes/mosaico00{}.jpg".format(i) )

for i in range(10, 12):
    mosaico_etsiit.append("imagenes/mosaico0{}.jpg".format(i) )

imagenes_etsiit = []

for imagen in mosaico_etsiit:
    imagenes_etsiit.append(leeimagen(imagen, 1))

print("\n Apartado 3 \n")

# hacemos el panorama de la etsiit y de las dos imagenes de yosemite
panorama_etsiit_3 = panorama_3_imagenes(*imagenes_etsiit[2:5])


mostrar_imagen(panorama_etsiit_3)
input("---------- Pulsa una tecla para continuar ----------")

print("\n Apartado 4 \n")

panorama_etsiit = panorama_imagenes(imagenes_etsiit)
panorama_yosemite = panorama_2_imagenes(yosemite_1_color, yosemite_2_color)

# mostramos los resultados
mostrar_imagen(panorama_yosemite)
input("---------- Pulsa una tecla para continuar ----------")
mostrar_imagen(panorama_etsiit)
input("---------- Pulsa una tecla para continuar ----------")



"""
Bonus 1:
"""

print("\n BONUS 1 \n")

def calcular_homografia(puntos_fuente, puntos_destino):
    # funcion para calcular una homografia a partir de dos listas de puntos

    valores_matriz = []
    i = 0
    for i in range(len(puntos_fuente)):
        p1 = np.array([puntos_fuente[i][0], puntos_fuente[i][1], 1])
        p2 = np.array([puntos_destino[i][0], puntos_destino[i][1], 1])

        a2 = [0, 0, 0, -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2],
              p2[1] * p1[0], p2[1] * p1[1], p2[1] * p1[2]]
        a1 = [-p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], 0, 0, 0,
              p2[0] * p1[0], p2[0] * p1[1], p2[0] * p1[2]]

        valores_matriz.append(a1)
        valores_matriz.append(a2)

    matriz = np.array(valores_matriz)

    u, s, v = np.linalg.svd(matriz)

    homografia = np.reshape(v[8], (3, 3))

    homografia = (1/homografia[2][2]) * homografia

    return homografia


def calcular_distancia(punto1, punto2, homografia):
    # calcular la distancia entre dos puntos relacionados por una homografia

    p1 = np.transpose(np.array([punto1[0], punto1[1], 1]) )
    estimacionp2 = np.dot(homografia, p1)
    estimacionp2 = (1/estimacionp2[2]) * estimacionp2

    p2 = np.transpose(np.array([punto2[0], punto2[1], 1]) )
    error = p2 - estimacionp2

    return np.linalg.norm(error)


def ransac(puntos_fuente, puntos_destino, umbral):

    # algoritmo ransac, con respecto al paper de la bibligrafia

    max_inliers = []
    homografia_final = np.zeros((3,3))

    j = 0

    # paramos si hemos dado las iteraciones dadas por el umbral, o si tenemos suficientes inliers
    while len(max_inliers) < len(puntos_fuente) * umbral and j < umbral:

        # 1: buscamos 1 correspondencias aleatorias
        cAleatorias = np.random.choice(len(puntos_fuente), 4, replace = False)

        # 2: calculamos la homografia
        homografia = calcular_homografia(puntos_fuente[cAleatorias], puntos_destino[cAleatorias])

        inliers = []

        # 3: aplicamos la homografia a todos los puntos
        for i in range(len(puntos_fuente)):
            distancia = calcular_distancia(puntos_fuente[i], puntos_destino[i], homografia)

            # 4: si la distancia es menor a un umbral, lo contamos con inlier
            if distancia < umbral:
                inliers.append([puntos_fuente[i], puntos_destino[i]])

        # si esta homografia produce mas inliers, actualizamos la homografia de resultado
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            homografia_final = np.copy(homografia)

        j +=1

    return homografia_final, max_inliers


def panorama_imagenes_ransac_propio(imagenes):

    """
    Funcion para crear un panorama entre N imagenes
    """

    # se supone que estan ordenadas de derecha a izquierda
    # centro sera la posicion donde esta en imagenes la imagen central
    centro = len(imagenes) // 2

    # calculamos el tamaño para el resultado
    # aqui nos ponemos en el peor de los casos y creamos el resultado
    # más grande de lo que deberíamos, luego lo ajustamos
    tam_resul_x = 0
    tam_resul_y = 0

    for imagen in imagenes:
        tam_resul_x += imagen.shape[1]
        tam_resul_y += imagen.shape[0]

    desp_homo_x = tam_resul_x // 2
    desp_homo_y = tam_resul_y // 2

    resultado = np.zeros( ( tam_resul_x, tam_resul_y ) , dtype = np.uint8 )
    resultado = cv.cvtColor(resultado, cv.COLOR_BGR2RGB)

    imagen_central = normaliza_imagen_255(imagenes[centro])

    # creamos la homografia inicial. Como vamos a comenzar con la imagen central
    # el desplazamiento es la mitad del cuadro de resultado generado, para
    # ponerla en el centro
    homografia = np.array( [[1, 0, desp_homo_x],
                            [0, 1, desp_homo_y],
                            [0, 0, 1]], dtype = np.float64 )

    # ponemos la imagen central
    resultado = cv.warpPerspective(imagen_central, homografia, (tam_resul_x, tam_resul_y), dst=resultado, borderMode = cv.BORDER_TRANSPARENT)


    # volvemos a la homografia inicial, ya que empezamos desde el inicio, pero
    # hacia la izquierda
    copia_homografia = np.copy(homografia)

    # parte izquierda del panorama
    for i in range(centro, 0, -1):
        destino = imagenes[i]
        fuente = imagenes[i - 1]

        # sacamos puntos de interes y descriptores
        puntos_interes_destino, descriptores_destino = puntos_descriptores_AKAZE(destino, 0.01)
        puntos_interes_fuente, descriptores_fuente = puntos_descriptores_AKAZE(fuente, 0.01)

        # sacamos coincidencias
        coincidencias = coincidencias_descriptores_lowe_average_2nn(descriptores_destino, descriptores_fuente)

        puntos_destino = []
        puntos_fuente = []

        # sacamos los puntos de las coincidencias
        for coincidencia in coincidencias:
            puntos_destino.append(puntos_interes_destino[coincidencia.queryIdx].pt)
            puntos_fuente.append(puntos_interes_fuente[coincidencia.trainIdx].pt)

        puntos_destino = np.array(puntos_destino, dtype = np.float32)
        puntos_fuente = np.array(puntos_fuente, dtype = np.float32)

        # al igual que antes, obtenemos la homografia
        homografia_cv, _ = ransac(puntos_fuente, puntos_destino, 5)
        # la apilamos con las anteriores
        copia_homografia = np.dot(copia_homografia, homografia_cv)

        copia_fuente = normaliza_imagen_255(fuente)

        # y la aplicamos
        resultado = cv.warpPerspective(copia_fuente, copia_homografia, (tam_resul_x, tam_resul_y), dst=resultado, borderMode = cv.BORDER_TRANSPARENT)

    # guardamos el extremo utilizado por la homografia, para recortar la imagen.
    # en este caso no necesitamos aplicar un ajuste ya que la imagen se ha colocado
    # en la derecha de este punto, y nos interesa el extremo izquierdo
    ancho_min = copia_homografia[1][2]
    alto_min = copia_homografia[0][2]

    ancho_min = int(ancho_min)
    alto_min = int(alto_min)



    # hacemos una copia de la homografia
    copia_homografia = np.copy(homografia)

    # parte derecha del panorama
    for i in range ( centro, len(imagenes) - 1 ):

        destino = imagenes[i]
        fuente = imagenes[i + 1]

        # sacamos los puntos y descriptores de cada imagen
        puntos_interes_destino, descriptores_destino = puntos_descriptores_AKAZE(destino, 0.01)
        puntos_interes_fuente, descriptores_fuente = puntos_descriptores_AKAZE(fuente, 0.01)

        # las coincidencias con lowe 2NN
        coincidencias = coincidencias_descriptores_lowe_average_2nn(descriptores_destino, descriptores_fuente)

        puntos_destino = []
        puntos_fuente = []

        # sacamos los puntos de destino y fuente donde hay coincidencias
        for coincidencia in coincidencias:
            puntos_destino.append(puntos_interes_destino[coincidencia.queryIdx].pt)
            puntos_fuente.append(puntos_interes_fuente[coincidencia.trainIdx].pt)

        puntos_destino = np.array(puntos_destino, dtype = np.float32)
        puntos_fuente = np.array(puntos_fuente, dtype = np.float32)


        # obtenemos la homografia donde iría la imagen
        homografia_cv, _ = ransac(puntos_fuente, puntos_destino, 5)
        # aplicamos una multiplicacion matricual para actualizar la homografia
        # que estamos usando, de cara a que se apilen las transformaciones
        # de forma que se vayan posicionando de forma correcta el resultado
        copia_homografia = np.dot(copia_homografia, homografia_cv)

        copia_fuente = normaliza_imagen_255(fuente)

        # por ultimo, añadimos la imagen al resultado con warpPerspective
        resultado = cv.warpPerspective(copia_fuente, copia_homografia, (tam_resul_x, tam_resul_y), dst=resultado, borderMode = cv.BORDER_TRANSPARENT)


    # cuando hacemos todas las de la parte derecha, sabemos hasta donde ha llegado
    # para recortar el resultado. Añadimos cierto margen, ya que al deformar las
    # imagenes y el propio tamaño de la imagen hace que necesitemos más espacio
    ancho_max = copia_homografia[1][2] + imagenes[len(imagenes) - 1].shape[0] * 1.4
    alto_max = copia_homografia[0][2] + imagenes[len(imagenes) - 1].shape[1] * 1.4

    # lo pasamos a entero
    ancho_max = int(ancho_max)
    alto_max = int(alto_max)

    # el resultado es el resultado original, pero recortando toda la zona negra sin utilizar
    resultado = resultado[ancho_min:ancho_max, alto_min:alto_max ]

    return resultado



# hacemos el panorama de la etsiit y de las dos imagenes de yosemite

panorama_yosemite_propio = panorama_imagenes_ransac_propio([yosemite_1_color, yosemite_2_color] )
mostrar_imagen(panorama_yosemite_propio)
mostrar_imagenes([panorama_yosemite, panorama_yosemite_propio])
input("---------- Pulsa una tecla para continuar ----------")


panorama_etsiit_ransac_propio = panorama_imagenes_ransac_propio(imagenes_etsiit)

# mostramos los resultados
mostrar_imagen(panorama_etsiit_ransac_propio)
mostrar_imagenes([panorama_etsiit, panorama_etsiit_ransac_propio])
input("---------- Pulsa una tecla para continuar ----------")


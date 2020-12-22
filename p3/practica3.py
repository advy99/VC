#!/usr/bin/env python
# -*- coding: utf-8 -*-



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


def aplicar_convolucion(imagen, k_x, k_y):

    k_x_invertido = np.flip(k_x)
    k_y_invertido = np.flip(k_y)

    img_conv_c = cv.filter2D(imagen, -1, k_x_invertido)
    img_conv_final = cv.filter2D(imagen, -1, k_y_invertido)

    return img_conv_final

def puntos_harris(imagen, tam_bloque):

    piramide_gauss = piramide_gaussiana_cv(imagen, 3)




"""
Apartado 1
"""

yosemite_1_bn = leeimagen("imagenes/Yosemite1.jpg", 0)
yosemite_2_bn = leeimagen("imagenes/Yosemite2.jpg", 0)



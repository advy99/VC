"""
VC: Práctica 0

25/09/2020

author: Antonio David Villegas Yeguas
"""

import matplotlib.pyplot as plt
import cv2 as cv

"""
 Ejercicio 1:


"""

"""
  lee_imagen:
    Leer un fichero como una imagen usando OpenCV.

  Argumentos:
    filename: Ruta a la imagen
    flag_color: Modo de color a leer (0: B/N, 1: Color)

"""
def lee_imagen (filename, flag_color=None):
  imagen = cv.imread(filename, flag_color)

  return imagen


def mostrar_imagen(titulo, imagen):
  cv.namedWindow(titulo, cv.WINDOW_NORMAL)
  cv.imshow(titulo, imagen)
  cv.waitKey(0)
  cv.destroyAllWindows()

  # Si OpenCV da error
  # plt.imshow(imagen)
  # plt.show()


filename = "images/orapple.jpg"


imagen_leida = lee_imagen(filename, 0)

mostrar_imagen("Imagen leida B/N", imagen_leida)


imagen_leida_color = lee_imagen(filename, 1)

mostrar_imagen("Imagen leida C", imagen_leida_color)

input("\n-------Pulsa una tecla para continuar-------\n")

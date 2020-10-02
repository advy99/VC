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
    fichero: Ruta a la imagen
    flag_color: Modo de color a leer (0: B/N, 1: Color)

"""
def lee_imagen_fichero (fichero, flag_color):
	imagen = cv.imread(fichero, flag_color)

	return imagen


def mostrar_imagen(imagen, titulo=""):
	cv.namedWindow(titulo, cv.WINDOW_NORMAL)
	cv.imshow(titulo, imagen)
	cv.waitKey(0)
	cv.destroyAllWindows()

	# Si OpenCV da error
	# plt.imshow(imagen)
	# plt.show()

def leeimagen(fichero, color_flag):
	imagen = lee_imagen_fichero(fichero, color_flag)
	mostrar_imagen(imagen)





print("Ejercicio 1:")


color_flag = 0
leeimagen("imagenes/orapple.jpg", color_flag)

color_flag = 1
leeimagen("imagenes/orapple.jpg", color_flag)


input("\n-------Pulsa una tecla para continuar-------\n")




print("Ejercicio 2:")

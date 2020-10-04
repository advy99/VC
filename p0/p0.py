"""
VC: Práctica 0

25/09/2020

author: Antonio David Villegas Yeguas
"""

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

np.random.seed(1)



"""
 Ejercicio 1:

Escribir una función que lea el fichero de una imagen permita mostrarlatanto
en grises como en color (im=leeimagen(filename, flagColor))

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



"""

Ejercicio 2:

Escribir una función que visualice una matriz de números reales cualquiera ya
sea monobanda o tribanda (pintaI(im)). Para ello deberá de trasladar y escalar
el rango de cada banda al intervalo [0,1] sin pérdida de información.


"""


print("Ejercicio 2:")


def pintaI(imagen):

	# trabajamos en una copia de la imagen
	img_normalizada = np.copy(imagen)

	# sacamos el minimo y el maximo de todos los valores
	minimo = np.min(img_normalizada)
	maximo = np.max(img_normalizada)

	img_normalizada = img_normalizada.astype(np.float64)

	# si es monobanda la convertimos a tribanda apilando dos veces en profundidad
	# la misma imagen
	if img_normalizada.ndim == 2:
		img_normalizada = np.dstack((np.copy(imagen), np.copy(imagen)))
		img_normalizada = np.dstack((img_normalizada, np.copy(imagen)))


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



#imagen = lee_imagen_fichero("imagenes/orapple.jpg", 1)

print("Probando con matriz aleatoria como imagen monobanda")
imagen_monobanda = np.random.randn(500,500,1) * 5

mostrar_imagen(pintaI(imagen_monobanda))

input("\n-------Pulsa una tecla para continuar-------\n")


print("Probando con matriz aleatoria como imagen tribanda")
imagen_tribanda = np.random.randn(500,500,3) * 5

mostrar_imagen(pintaI(imagen_tribanda))

input("\n-------Pulsa una tecla para continuar-------\n")

print('Probando con una de las imagenes dadas')
imagen = lee_imagen_fichero("imagenes/orapple.jpg",1)

mostrar_imagen(pintaI(imagen))
input("\n-------Pulsa una tecla para continuar-------\n")




"""
Ejercicio 3:

Escribir una función que visualice varias imágenes a la vez(fusionando las
imágenes en una última imagen final): pintaMI(vim).(vim será una secuencia de
imágenes) ¿Qué pasa si las imágenes no son todas del mismo tipo(nivel de gris,
color, blanco-negro)?

"""

print("Ejercicio 3")


def pintaIM(vim):

	num_imagenes = len(vim)

	alturas = []

	for i in range(0, num_imagenes):
		alturas.append( vim[i].shape[0])

	altura_maxima = np.max( alturas )

	# cogemos la primera imagen normalizada
	imagen_final = pintaI(vim[0])

	if imagen_final.shape[0] < altura_maxima:
		filas_restantes = altura_maxima - vim[0].shape[0]
		franja_negra = np.ones( (filas_restantes, vim[0].shape[1]))
		franja_negra = pintaI(franja_negra)
		imagen_final = np.vstack((imagen_final, franja_negra ))


	for i in range(1, num_imagenes):
		# para las siguientes imagenes, las normalizamos
		img = pintaI(vim[i])

		# si les faltan filas, añadimos las restantes como un borde negro
		if img.shape[0] < altura_maxima:
			filas_restantes = altura_maxima - img.shape[0]
			franja_negra = np.ones( (filas_restantes, img.shape[1]))
			franja_negra = pintaI(franja_negra)
			img = np.vstack((img, franja_negra ))

		imagen_final = np.hstack((imagen_final, img))

	mostrar_imagen(imagen_final)


imagen1 = lee_imagen_fichero("imagenes/orapple.jpg", 1)
imagen2 = lee_imagen_fichero("imagenes/messi.jpg", 1)

imagenes = [imagen2, imagen1]

pintaIM(imagenes)



"""
Ejercicio 4:

Escribir unafunción que modifique el coloren la imagen de cada uno de los
elementos de una lista de coordenadas de píxeles. (Recordarque (fila, columna)
es lo contrario a (x,y). Es decir fila=y, columna=x)

"""


print("Ejercicio 4:")


def modificar_color(imagen, coordenadas_a_modificar, color):
	resultado = np.copy(imagen)

	for coordenada in coordenadas_a_modificar:
		x, y = coordenada

		# y, x en lugar de x, y, porque en las imagenes las filas son las
		# columnas de la matriz
		resultado[y, x] = nuevo_color

	return resultado

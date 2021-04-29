import numpy as np

def caracteristicas_del_array():
    aa = np.array([1, 2, 3, 4], dtype=np.int8)
    bb = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    print("longitud de una dimencion\n", aa.shape)
    print("longitud de cada dimencion", bb.shape)  # The shape of bb
    print("numero de dimenciones", bb.ndim)  # The dimensions of bb
    print("reordenar matriz\n", aa[np.array([3, 1, 2, -4])])
    print("elementos totales del array:", bb.size)  # b has 9 elements
    print("los bytes que ocupa un item:", bb.itemsize)  # The size of element in bb.
    print("elementos x bytes:", bb.nbytes)  # check how many bytes in bb.
    #bb.resize((2, 1))
    #print(bb)

def actualizar_valores():
    arr = np.asarray([[1, 3, 5], [7, 9, 6]], dtype=np.uint8)
    print("array simple")
    print(arr)
    print("actualizar pocicion", )
    arr[1, 2] = 66
    print(arr)
    print("actualizar pocicion item", )
    arr.itemset((0, 2), 55)
    print(arr)
    print("actualizar por orden absoluto",)
    arr.put([0,3,5],[0,0,0])
    print(arr)
    print("tipo de array", arr.dtype)
    arr.astype('float32')
    print("tipo de array", arr.dtype)

def leer_valores():
    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                  [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                  [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                  [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]],
                 dtype='uint8')  # .astype('float32')
    print("extraer pocicion", a[4, 2])
    print("extraer pocicion por item", a.item(4, 2))
    print("extraer pocicion por cordenada", a[4][2])  #
    print("filas", a[[1, 2]])  # fila 2 y 3
    print('fila', a[0, :])  # fila
    print("cuadrante", a[2:4, 0:2])
    print('columa', a[:, 2])  # columna
    print("elementos segun el orden absoluto", np.take(a, [3, 9, 10, 15, 25, 30]))
    print("primeras 3 filas y saltar de 2 en 2:\n", a[:3, ::3])
    print("a[1:2 ,[1,2]] = (rama 1 (la 2 no) de la primera dimencion, items 1 y 2 )\n", a[1:2, [1, 2]])
    print("a[0, 3:5]  = (rama 0 de la primera dimencion, del 3 al 4 (el 5 no) \n", a[0, 3:5])
    print("a[4:, 4:]  = (del 4 en delante , del 4 en delane) \n", a[4:, 4:])
    print("a[:, 2]  = (todas las ramas de la primera dimencion , items pocicion 2) \n", a[:, 2])
    print("a[2::2, ::2]  = (despues del 2 :: cada 2, todos : cada 2) \n", a[2::2, ::2])
    # este sub_array diagonal no es intermediario
    print('diagonal por interseccion de cordenadas', a[[0, 1, 2, 3, 4], [3, 4, 5, 6, 7]])  # interseccion de cordenadas
    print("Exteaer diagonal: \n", np.diag(a), "\n")
    print("Exteaer diagonal recorrida -1: \n", np.diag(a,-1), "\n")
    print("Exteaer diagonal recorrida 1: \n", np.diag(a, 1), "\n")

def matris_intermediara():
    print("matris intermediaria")  # se pude tomar un segmento de de matris y usarlo como intermediario para editar el elemento principal
    x = np.array([[1, 2, 3], [4, 5, 6]], np.int8)
    print("matris original\n",x)
    y = x[:, 1]
    y[0] = 9  # this also changes the corresponding element in x
    print("sub matris intermediaria\n",y)
    print("matris original midificada indirectamente\n",x)

def comparacion_de_matrices():
    rand = np.random.randint(0, 100, size=(4, 4), dtype='uint8')  # rand(3,4)
    print("random espesif", np.random.choice([1, 3, 5], p=[0.2, 0.3, 0.5], size=8))
    print(rand)
    print("matris de comparacion (mayor o menor que)")
    print((rand > 33) & (rand < 66))
    print(~((rand > 33) & (rand < 66)))
    print("mayor que 50")
    print(rand[rand > 50])

def tipos_de_rangos():
    print("rango:\n", np.arange(1, 10))
    print("rango inverso:\n", np.arange(10, 1, -2))
    print("rango_linspace:\n", np.linspace(1.0, 9.0, num=17))
    print("rango_logaritmico:\n", np.geomspace(1, 256, 9))# , endpoint=False)
    print("rango escala dividida", np.linspace(0, 12, 24))

def creacion_de_matrices():
    z = np.ones((3, 3), dtype='uint8')
    z_copia = np.ones_like(z)  # mismas dimenciones y tipo de datos
    x = np.zeros((3, 3), dtype='uint8')
    x_copia = np.zeros_like(x)  # mismas dimenciones y tipo de datos
    n = np.full((3, 3), 45, dtype='uint8')
    # n_copia = np.full_like(z)# mismas dimenciones y tipo de datos
    copya_simple = np.copy(z)
    print("copya_simple: \n", copya_simple)
    print("ones:\n", z)
    print("zeros:\n", x)
    print("full:\n", n)
    print("stak de matrices:\n", np.vstack(([z], [x], [n])))# marge()
    print("crear y doblar:\n", np.arange(27).reshape((3, 3, 3)))
    my_tuple = ([1, 3, 9], [8, 2, 6])
    my_list = [[1, 3, 5], [7, 9, 6]]
    arr = np.asarray(my_list, dtype=np.uint8)  # convertir tupla o lista a matriz


def crear_diagonal():
    print("cuadrado:\n", np.eye(3))
    print("recrangulo:\n", np.eye(7, 6, -3))
    print("cuadrado diagonal recorrida:\n", np.eye(6, k=-1 ))
    print("triangulo:\n",np.tri(3))
    print("triangulo de diagonal recorrida:\n", np.tri(4, k=1))
    # print("recortar triangulo inverso:\n", np.triunfo(a))
    print("crear diagonal:\n", np.diagflat([1, 7, 6]), "\n")
    print("crear diagonal recorrida:\n", np.diagflat([1, 7, 6], 1), "\n")

def algebra_de_arrays():
    arrayA = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arrayB = np.array([10, 20, 30])
    print("suma en dimencion 2:\n", arrayA + arrayB)
    arrayD = np.array([[10], [20], [30]])
    print("suma en dimencion 1:\n", arrayA + arrayD )

    arr1 = np.array([[4, 7], [2, 6]], dtype=np.float64)
    arr2 = np.array([[3, 6], [2, 8]], dtype=np.float64)

    print("Addition of Two Arrays: ")
    print(np.add(arr1, arr2))
    print(arr1 + arr2)
    print("subtract of Two Arrays: ")
    print(np.subtract(arr1, arr2))
    print(arr1 - arr2)
    print("mult  Two Arrays: ")
    print(np.multiply(arr1, arr2))
    print(arr1 * arr2)
    print("dividir Two Arrays: ")
    print(np.divide(arr1, arr2))
    print(arr1 / arr2)

    print("\nsuma total del Array:\n", np.sum(arrayA))
    print("\nsuma entre columnas:\n", arrayA.sum(0))
    print("\nsuma algebraica de filas:\n", arrayA.sum(-1))
    print("\nRaiz ciadrada: \n", np.sqrt(arrayA))

def rehacer_forma_y_extender_anadir():
    arrayA = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arrayB = np.array([10, 20, 30])
    arrayC = np.arange(8)
    print("array base:\n", arrayC)
    print("reshape 2,4:\n", np.reshape(arrayC, (2, 4)))
    print("reshape 4,2:\n", np.reshape(arrayC, (4, 2)))
    print("reshape 8,1:\n", arrayC.reshape(8, 1))# reshap estructura 2
    print("reshape 2,2,2:\n", arrayC.reshape(2, 2, 2))
    print("reshape de 3x3 a 1x9:\n", np.reshape(arrayA, (1, 9)))
    print("hacer una dimencion y añadir unidemencionalmente:\n",
          np.append(arrayA, arrayC))  # se conviertenen un array de una dimencion y se suman
    print("añadir/extender filas o columnas:\n", np.append(arrayA, arrayA, axis=0))
    # se conviertenen un array de una dimencion y se suman
    print("unidimecionar y concatenar:\n", np.concatenate((arrayA.flatten(), arrayC.flatten())))
    print("convertir a array a una dimencion:\n", arrayA.flatten())
    print("palindromo, invertir orden", arrayB[::-1])

def tipos_de_datos_numericos():
    int_8 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    int_8 = int_8 + 128
    print("sumar 128 a int8 sobrepasa a int16 un int8"
          " es desde -128 hasta 128\n", int_8.dtype)
    int8_positivo = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    int8_positivo = int8_positivo + 256
    print("sumar 255 a uint8 sobrepasa a uint16 un uint8"
          " es desde 0 hasta 256\n", int8_positivo.dtype)

def algebra():
    a = np.array([1, 2, 3, 4, 5, 6], dtype=np.int8).reshape(2, 3)
    print("orden nornal A:\n", a)
    print("transpuesta:\n", a.T)
    print("concatenacion de listas")
    print(np.identity(4, dtype='uint8'))


def siclos_logicos():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    array_t = array.T

    print("iterador simplel:")
    for x in np.nditer(array):
        print(x)

    print('el iterador se ejecuta segun la variable original y no seguan la intermediaria:')
    for x in np.nditer(array_t):
        print(x)
    print('el iterador se ejecuta en orden seguan la variable intermediaria :')
    for x in np.nditer(array_t, order='C'):
        print(x)
    print('el iterador se ejecuta segun la variable original :')
    for x in np.nditer(array_t, order='F'):
        print(x)

    print("external_loop")
    for x in np.nditer(array, flags=['external_loop'], order='C'):
        print(x)
    print('Modified array is:')
    print(array)

    print('iterador con id')
    it = np.nditer(array, flags=['f_index'])
    while not it.finished:
        print("%d - %d id" % (it[0], it.index), end="\n")
        it.iternext()

    array_2 = np.array([1, 2, 3], dtype=int)
    print("iterable de arrays multiple desiguales")
    for x, y in np.nditer([array, array_2]):
        print("%d-%d   " % (x, y))

    print("iterable de arrays multiple multidimencional")
    array_2 = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]])
    print('a-b')
    for (i, j) in zip(array, array_2):
        for (h, y) in zip(i, j):
            print(h, ' ', y)

    print('editar cada elemento del array con ciclos')
    for x in np.nditer(array, op_flags=['readwrite']):#'readwrite' o 'writeonly'
        x[...] = 5 * x
    print(array)

    #print('nditer con with ')
    #with np.nditer(array, op_flags=['readwrite']) as it:
    #    for x in it:
    #        x[...] = 2 * x


def orden_de_array():
    desorden = np.array([[3, 2], [5, 1], [8, 6], [4, 7]])
    print("desorden:\n", desorden)
    print("ordenar/reorganiza de columna en columna : \n", np.sort(desorden, axis=0))
    print("ordena/reorganiza de fila en fila : \n", np.sort(desorden, axis=-1))
    print("ordena/reorganiza linealmente : \n", np.sort(desorden, axis=None))
    print("countador de no zeros : ", np.count_nonzero([[0, 1, 7, 0, 4], [3, 0, 0, 2, 19]]))

def comparar_arrays():
    a1 = np.array([1, 2, 4, 6, 7])
    a2 = np.array([1, 3, 4, 5, 7])
    a3 = np.array([1, 3, 4.00001, 5, 7])

    print("-")
    print(np.array_equal(a1, a1))
    print(np.array_equal(a1, a2))
    print("-")

    print(np.allclose(a1, a2))
    print(np.allclose(a3, a2))
    print("-")

    print(np.array_equiv(a1, a2))
    print(np.array_equiv(a3, a2))
    print("-")

    print((a1 == a2).all())
    print((a3 == a2).all())

#caracteristicas_del_array()
#actualizar_valores()
#leer_valores()
#matris_intermediara()
#comparacion_de_matrices()
#creacion_de_matrices()
#crear_diagonal()
#algebra_de_arrays()
#rehacer_forma_y_extender_anadir()
#tipos_de_datos_numericos()
#algebra()
#siclos_logicos()
#orden_de_array()


#operaciones con los bits de los numeros del array
#numpy.bitwise_and()
#numpy.bitwise_and()
#numpy.bitwise_xor()
#numpy.invert()
#numpy.left_shift()
#numpy.right_shift()
#numpy.binary_repr(number, width=None)



'''
ndarray.flags       Información sobre el diseño de la memoria de la matriz.
ndarray.shape       Tupla de dimensiones de matriz.
ndarray.strides     Tupla de bytes para avanzar en cada dimensión al atravesar una matriz.
ndarray.ndim        Número de dimensiones de la matriz.
ndarray.data        Objeto de búfer de Python que apunta al inicio de los datos de la matriz.
ndarray.size        Número de elementos de la matriz.
ndarray.itemsize    Longitud de un elemento de la matriz en bytes.
ndarray.nbytes      Total de bytes consumidos por los elementos de la matriz.
ndarray.base        Objeto base si la memoria es de algún otro objeto.

ndarray.T           La matriz transpuesta.
ndarray.real        La parte real de la matriz.
ndarray.imag        La parte imaginaria de la matriz.
ndarray.flat        Un iterador 1-D sobre la matriz.
ndarray.ctypes      Un objeto para simplificar la interacción de la matriz con el módulo ctypes.
'''



import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def cruzado_compuertas_logicas():
    blank = np.zeros((200, 200), dtype='uint8')

    rectangle = cv.rectangle(blank.copy(), (15, 15), (185, 185), 255, -1)
    circle = cv.circle(blank.copy(), (100, 100), 100, 255, -1)

    cv.imshow('Rectangle', rectangle)
    cv.imshow('Circle', circle)

    bitwise_and = cv.bitwise_and(rectangle, circle)
    cv.imshow('AND', bitwise_and)

    bitwise_or = cv.bitwise_or(rectangle, circle)
    cv.imshow('OR', bitwise_or)

    bitwise_xor = cv.bitwise_xor(rectangle, circle)
    cv.imshow('XOR', bitwise_xor)

    bitwise_nand = cv.bitwise_not(bitwise_and)
    cv.imshow('NAND', bitwise_nand)

    bitwise_not = cv.bitwise_not(circle)
    cv.imshow('NOT', bitwise_not)

    cv.waitKey(0)
    cv.destroyAllWindows()

def video_play():
    capture = cv.VideoCapture('assets/figuras.mp4')
    while True:
        isTrue, fream = capture.read()
        #fream = cv.cvtColor(fream, cv.COLOR_BGR2GRAY)
        if isTrue:
            #fream = rescale_img(fream, 0.5)
            cv.imshow('video', fream)
        else:
            break
        # 1 milisegundo entre pantallas esperando, cerrar con 'd'
        if cv.waitKey(1) & 0xFF == ord('a'):
            break
    capture.release()
    cv.waitKey(0)
    cv.destroyAllWindows()

def recortar_imagen():
    img = cv.imread('assets/img_1.png')
    cv.imshow('Cats', img)
    blank = np.zeros(img.shape[:2], dtype='uint8')
    mask = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)
    masked = cv.bitwise_and(img, img, mask=mask)
    cv.imshow('Mask', masked)

def histograma():

    img = cv.imread('assets/img_1.png')
    cv.imshow('Cats', img)

    plt.figure()
    plt.title('Colour Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    colors = ('b', 'g', 'r')

    blank = np.zeros(img.shape[:2], dtype='uint8')
    mask = cv.rectangle(blank, (0,0),(640, 428), 255, -1)

    for i, col in enumerate(colors):
        hist = cv.calcHist([img], [i], mask , [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])

    blank = np.zeros(img.shape[:2], dtype='uint8')
    b, g, r = cv.split(img)

    blue = cv.merge([b, blank, blank])
    green = cv.merge([blank, g, blank])
    red = cv.merge([blank, blank, r])

    cv.imshow('Blue', blue)
    cv.imshow('Green', green)
    cv.imshow('Red', red)

    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()

def monocromatismo():
    img = cv.imread('assets/img_1.png')
    cv.imshow('Cats', img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshow, thres = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
    cv.imshow('corte monocromatico', thres)
    threshow, thres = cv.threshold(img, 100, 255, cv.THRESH_BINARY_INV)
    cv.imshow('corte monocromatico inverso', thres)
    adaptative_thres = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
    cv.imshow('corte monocromatico adaptativo', adaptative_thres)
    adaptative_gaus = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
    cv.imshow('corte monocromatico gause', adaptative_gaus)
    cv.waitKey(0)
    cv.destroyAllWindows()

def edge():
    img = cv.imread('assets/park.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Laplacian', np.uint8(np.absolute(cv.Laplacian(img, cv.CV_64F))))
    cv.imshow('Sobel X', cv.Sobel(img, cv.CV_64F, 1, 0))
    cv.imshow('Sobel Y', cv.Sobel(img, cv.CV_64F, 0, 1))
    cv.imshow('Combined Sobel Sobel X y Y', cv.bitwise_or(cv.Sobel(img, cv.CV_64F, 1, 1), cv.Sobel(img, cv.CV_64F, 0, 1)))
    cv.imshow('Canny', cv.Canny(img, 150, 175))
    cv.waitKey(0)
    cv.destroyAllWindows()

def matris_simple():
    blank = np.zeros((200, 200, 3), dtype='uint8')
    print("asignar colores a una matris")
    blank[:] = 0,255,0
    #blank[:] = 255,0,0#blue
    blank[10:40, 10:40] = 0, 0, 255
    cv.rectangle(blank,(50,10),(80,40), (0,0,255),thickness=0)#grosor interno y externo
    cv.line(blank,(10,50),(80,70), (0,0,255))
    cv.imshow('Green', blank)
    cv.waitKey(0)
    cv.destroyAllWindows()

def matris_simple_2():

    blank = np.zeros((500, 500, 3), dtype='uint8')

    blank[200:300, 300:400] = 0, 0, 255

    cv.rectangle(blank, (0, 0), (blank.shape[1] // 2, blank.shape[0] // 2), (0, 255, 0), thickness=-1)
    cv.rectangle(blank, (30, 30), (60, 60), (0, 0, 255), thickness=0)  # grosor interno y externo
    cv.imshow('Rectangle', blank)

    cv.circle(blank, (blank.shape[1] // 2, blank.shape[0] // 2), 50, (0, 0, 255), thickness=-1)
    cv.imshow('Circle', blank)

    cv.line(blank, (100, 250), (300, 400), (255, 255, 255), thickness=3)
    cv.imshow('Line', blank)

    cv.putText(blank, 'gus', (0, 100), cv.FONT_ITALIC, 1.0, (255, 0, 0), 1)
    cv.imshow('Text', blank)
    cv.waitKey(0)
    cv.destroyAllWindows()

def reducir_bordes():
    img = cv.imread('assets/img_1.png')  #
    #img = rescale_img(img, 0.5)
    #cv.imshow('original', img)
    print("ajustar a color o gris")
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB) #color orden RGB
    print("atenuar bordes haciendo borroso")
    #img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT) #suabisador de bordes
    #img = cv.blur(img, (5, 5))
    #img = cv.medianBlur(img, 3)#suabisar similares y matrcar bordes
    #img = cv.dilate(img, (7, 7), iterations=5)
    img = cv.erode(img, (7, 7), iterations=3)
    cv.imshow('imagen borrosa', img)

    cv.imshow('canny bordes principales', cv.Canny(img, 200,255))
    cv.imshow('canny todos bordes', cv.Canny(img, 0, 50))
    cv.waitKey(0)
    cv.destroyAllWindows()

def transformaciones():

    def rescale_img(frame, scale=0.75):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    def translate(img, x, y):
        transMat = np.float32([[1, 0, x], [0, 1, y]])
        dimensions = (img.shape[1], img.shape[0])
        return cv.warpAffine(img, transMat, dimensions)

    def rotate(img, angle, rotPoint=None):
        (height, width) = img.shape[:2]

        if rotPoint is None:
            rotPoint = (width // 2, height // 2)

        rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
        dimensions = (width, height)

        return cv.warpAffine(img, rotMat, dimensions)

    img = cv.imread('assets/img_1.png')  #
    img = rescale_img(img, 0.5)
    cv.imshow('Resize', cv.resize(img, (300, 500), interpolation=cv.INTER_CUBIC))
    cv.imshow('Rotated', rotate(img, -45))
    cv.imshow('Translated', translate(img, -100, 100))
    cv.imshow('figuras flip', cv.flip(img, 0))

    cv.waitKey(0)
    cv.destroyAllWindows()

def unir_colores():
    img = cv.imread('assets/img_1.png')
    cv.imshow('Cats', img)

    b, g, r = cv.split(img)

    blank = np.zeros(img.shape[:2], dtype='uint8')

    cv.imshow('Blue', cv.merge([b, blank, blank]))
    cv.imshow('Green', cv.merge([blank, g, blank]))
    cv.imshow('Red', cv.merge([blank, blank, r]))

    merged = cv.merge([b, g, r])
    cv.imshow('Merged Image', merged)

    cv.waitKey(0)
    cv.destroyAllWindows()

def filtro_desuabizado():
    img = cv.imread('assets/img_1.png')
    cv.imshow('A1', cv.bilateralFilter(img, 1, 99, 99))
    cv.imshow('A2', cv.bilateralFilter(img, 99, 99, 1))
    cv.imshow('A3', cv.bilateralFilter(img, 51, 51, 99))
    cv.imshow('A4', cv.bilateralFilter(img, 19, 50, 19))
    cv.imshow('A5', cv.bilateralFilter(img, 51, 99, 1))
    cv.imshow('A6', cv.bilateralFilter(img, 19, 99, 19))
    cv.imshow('A7', cv.bilateralFilter(img, 1, 99, 51))

    cv.imshow('A8', cv.bilateralFilter(img, 99, 99, 99))
    cv.imshow('A9', cv.bilateralFilter(img, 51, 99, 51))
    cv.imshow('A10', cv.bilateralFilter(img, 51, 99, 99))
    cv.imshow('A11', cv.bilateralFilter(img, 99, 51, 99))
    cv.imshow('A12', cv.bilateralFilter(img, 99, 51, 51))

    cv.waitKey(0)
    cv.destroyAllWindows()

def leer_imagen():
    img = cv.imread('assets/img.png', 0)
    cv.imshow('5 segundos Cats', img)
    img_color = cv.imread('assets/img.png')
    cv.imshow('5 segundos color', img_color)
    cv.waitKey(5000)
    cv.destroyAllWindows()

def guardar_imagen():
    import os

    directory = os.getcwd()
    print("selecconar un directorio de destino")

    img = cv.imread('assets/img.png')
    img_edge = cv.Canny(img, 150, 175)
    cv.imshow('Canny', img_edge)

    print("python se mureve a la ruta para guardar")
    os.chdir(directory)
    print("Before saving image:")
    print(os.listdir(directory))

    cv.imwrite('savedImage.jpg', img_edge)

    print("After saving image:")
    print(os.listdir(directory))

    print('Successfully saved')

    cv.waitKey(0)
    cv.destroyAllWindows()

def cruze_de_imagen():
    image1 = cv.imread('assets/img_1.png ')
    image2 = cv.imread('assets/img.png')
    image1 = cv.resize(image1, (640, 480), interpolation=cv.INTER_CUBIC)
    image2 = cv.resize(image2, (640, 480), interpolation=cv.INTER_CUBIC)

    weightedSum = cv.addWeighted(image1, 0.5, image2, 0.4, 0)
    cv.imshow('fucion Image', weightedSum)

    sub = cv.subtract(image1, image2)
    cv.imshow('Subtracted Image', sub)

    cv.waitKey(0)
    cv.destroyAllWindows()




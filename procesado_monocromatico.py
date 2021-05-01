import cv2 as cv
import numpy as np

def compuertas_logicas():
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

def transformaciones_morfologicas():

    ka,kb = 5,5

    img = cv.imread('assets/img_11.png ')
    #cv.imshow('img', img)
    #kernel = np.ones((ka,kb), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ka,kb))

    erosion = cv.erode(img, kernel)
    #cv.imshow('reduce lo blanco', erosion)

    dilation = cv.dilate(img, kernel)
    cv.imshow('aumenta lo blanco', dilation)

    gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
    cv.imshow('diferencia entre dilatacion y erocion', gradient)

    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    cv.imshow('reduce ruido blanco - (reduce y aumenta lo blanco)', opening)

    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    cv.imshow('reduce ruido negro (aumenta y reduce lo blanco)', closing)

    tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    cv.imshow('diferencia entre la reduccion de ruido blanco', tophat)

    blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
    cv.imshow('diferencia entre la reduccion de ruido negro', blackhat)

    cv.waitKey(0)
    cv.destroyAllWindows()

def monocromatismo_y_somras():

    par1, par2 = 175, 255

    img = cv.imread('assets/img_1.png')
    cv.imshow('Cats', img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshow, thres = cv.threshold(img,par1, par2, cv.THRESH_BINARY)
    cv.imshow('corte monocromatico', thres)
    threshow, thres = cv.threshold(img, par1, par2, cv.THRESH_BINARY_INV)
    cv.imshow('corte monocromatico inverso', thres)

    threshow_zero, thres_zero = cv.threshold(img,par1, par2, cv.THRESH_TOZERO)
    cv.imshow('corte monocromatico thres_zero', thres_zero)
    threshow_zero, thres_zero = cv.threshold(img, par1, par2, cv.THRESH_TOZERO_INV)
    cv.imshow('corte monocromatico inverso thres_zero', thres_zero)

    ret, thresh1 = cv.threshold(img, par1, par2, cv.THRESH_TRUNC)
    cv.imshow('contraste raro', thresh1)
    ret, thresh2 = cv.threshold(img, par1, par2, cv.THRESH_TOZERO)
    cv.imshow('quitar negros', thresh2)
    ret, thresh3 = cv.threshold(img, par1, par2, cv.THRESH_TOZERO_INV)
    cv.imshow('quitar blancos', thresh3)

    adaptative_thres = cv.adaptiveThreshold(img, par1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
    cv.imshow('corte monocromatico adaptativo', adaptative_thres)
    adaptative_gaus = cv.adaptiveThreshold(img, par1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
    cv.imshow('corte monocromatico gause', adaptative_gaus)

    ret, thresh_o = cv.threshold(img, par1, par2, cv.THRESH_OTSU)
    cv.imshow('dividir fondo y objetos', thresh_o)

    cv.waitKey(0)
    cv.destroyAllWindows()

def monocromatismo_edge():
    img = cv.imread('assets/park.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Laplacian', np.uint8(np.absolute(cv.Laplacian(img, cv.CV_64F))))
    cv.imshow('Sobel X', cv.Sobel(img, cv.CV_64F, 1, 0))
    cv.imshow('Sobel Y', cv.Sobel(img, cv.CV_64F, 0, 1))
    cv.imshow('Combined Sobel Sobel X y Y', cv.bitwise_or(cv.Sobel(img, cv.CV_64F, 1, 1), cv.Sobel(img, cv.CV_64F, 0, 1)))
    cv.imshow('Canny', cv.Canny(img, 150, 175))
    cv.waitKey(0)
    cv.destroyAllWindows()

def dilatacion_y_contracion_de_claro_y_oscuro():
    image = cv.imread('assets/img_1.png ')

    kernel5 = np.ones((5, 5), np.uint8)
    kernel6 = np.ones((6, 6), np.uint8)
    kernel9 = np.ones((9, 9), np.uint8)

    image5 = cv.erode(image, kernel5)
    cv.imshow('Image erode 5', image5)
    print("a")

    image6 = cv.erode(image, kernel6, cv.BORDER_REFLECT)
    cv.imshow('Image erode 6', image6)

    image9 = cv.erode(image, kernel9, cv.BORDER_REFLECT)
    cv.imshow('Image erode 7', image9)

    img_dilation5 = cv.dilate(image, kernel5, iterations=1)
    cv.imshow('Image dilate 5', img_dilation5)

    img_dilation6 = cv.dilate(image, kernel6, iterations=1)
    cv.imshow('Image dilate 6', img_dilation6)

    img_dilation9 = cv.dilate(image, kernel9, iterations=1)
    cv.imshow('Image dilate 9', img_dilation9)

    cv.waitKey(0)
    cv.destroyAllWindows()

def monocromatismo_erocion_vertical_horizontal():
    import cv2 as cv
    img = cv.imread('assets/img_3.png', 0)
    laplacian = cv.Laplacian(img, cv.CV_64F)
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

    cv.imshow('laplacian', laplacian)
    cv.imshow('sobelx', sobelx)
    cv.imshow('sobely', sobely)

    sobelx8u = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=5)
    sobelx64f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)

    cv.imshow('sobelx8u', sobelx8u)
    cv.imshow('sobelx64f', sobelx64f)
    cv.imshow('abs_sobel64f', abs_sobel64f)
    cv.imshow('sobel_8u', sobel_8u)

    cv.waitKey(0)
    cv.destroyAllWindows()

def quitar_sombra_binaria_inteligente_gauss():

    #img = cv.imread('assets/img_14.png', 0)
    #img = cv.imread('assets/img_20.png', 0)
    img = cv.imread('assets/img_24.png', 0)
    #img = cv.medianBlur(img, 5)
    ret, th1 = cv.threshold(img, 120, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        cv.imshow(titles[i], images[i])
    cv.waitKey(0)
    cv.destroyAllWindows()()


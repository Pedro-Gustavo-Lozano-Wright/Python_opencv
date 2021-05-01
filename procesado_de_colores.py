
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def histograma():

    img = cv.imread('assets/img_1.png')
    cv.imshow('Cats', img)

    plt.figure()
    plt.title('Colour Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    colors = ('b', 'g', 'r')

    #blank = np.zeros(img.shape[:2], dtype='uint8')
    #mask = cv.rectangle(blank, (0,0),(640, 428), 255, -1)


    # for i, col in enumerate(colors):
    #     hist = cv.calcHist([img], [i], mask , [256], [0, 256])
    #     plt.plot(hist, color=col)
    #     plt.xlim([0, 256])

    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist, color='b')
    plt.xlim([0, 256])

    hist = cv.calcHist([img], [1], None, [256], [0, 256])
    plt.plot(hist, color='g')
    plt.xlim([0, 256])

    hist = cv.calcHist([img], [2], None, [256], [0, 256])
    plt.plot(hist, color='r')
    plt.xlim([0, 256])

    print("suma de todos los colores")
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

    blank = np.zeros(img.shape[:2], dtype='uint8')
    b, g, r = cv.split(img)

    blue = cv.merge([b, blank, blank])
    green = cv.merge([blank, g, blank])
    red = cv.merge([blank, blank, r])

    #cv.imshow('Blue', blue)
    #cv.imshow('Green', green)
    #cv.imshow('Red', red)

    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()

def dibujo_simple():
    blank = np.zeros((200, 200, 3), dtype='uint8')

    print("asignar colores a una matris")
    blank[:] = 0,255,0
    #blank[:] = 255,0,0#blue
    cv.circle(blank, (blank.shape[1] // 2, blank.shape[0] // 2), 50, (0, 255, 255),-1)
    cv.rectangle(blank, (0, 0), (blank.shape[1] // 2, blank.shape[0] // 2), (255, 255, 0), thickness=-1)
    blank[10:40, 10:40] = 0, 0, 255
    cv.rectangle(blank,(50,10),(80,40), (0,0,255),thickness=0)#grosor interno y externo
    cv.line(blank,(10,50),(80,70), (0,0,255))
    blank = cv.arrowedLine(blank,(100,50),(180,70),(0,0,255))
    cv.putText(blank, 'gus', (0, 150), cv.FONT_ITALIC, 1.0, (255, 0, 0), 1)
    angle = 30
    startAngle = 0
    endAngle = 360
    blank = cv.ellipse(blank, (100,100), (30, 10), angle,
                        startAngle, endAngle, (255, 0, 0), -1)
    blank = cv.ellipse(blank, (150,170), (40, 10), angle,
                        startAngle, endAngle, (255, 0, 0))

    cv.imshow('Green', blank)
    cv.waitKey(0)
    cv.destroyAllWindows()

def dibujar_triangulo():

    image = cv.imread('assets/img_5.png')

    p1 = (180, 90)
    p2 = (120, 130)
    p3 = (110, 80)

    cv.line(image, p1, p2, (255, 0, 0), 3)
    cv.line(image, p2, p3, (255, 0, 0), 3)
    cv.line(image, p1, p3, (255, 0, 0), 3)

    centroid = ((p1[0] + p2[0] + p3[0]) // 3, (p1[1] + p2[1] + p3[1]) // 3)

    cv.circle(image, centroid, 4, (0, 0, 255))

    cv.imshow("image", image)
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

def suma_y_resta_de_imagen():

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

def dufuminar():

    image = cv.imread('assets/img_1.png ')
    cv.imshow('Original Image', image)

    Gaussian = cv.GaussianBlur(image, (7, 7), 0)
    cv.imshow('Gaussian Blurring', Gaussian)

    median = cv.medianBlur(image, 5)
    cv.imshow('Median Blurring', median)

    bilateral = cv.bilateralFilter(image, 9, 75, 75)
    cv.imshow('Bilateral Blurring', bilateral)

    cv.waitKey(0)
    cv.destroyAllWindows()

def aclarar_oscurecer_y_contrastar():

    img = cv.imread('assets/park.jpg')
    cv.imshow('Original Image', img)

    for gamma in [0.1, 0.5, 1.2, 2.2]:
        gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
        #cv.imwrite('gamma_transformed' + str(gamma) + '.jpg', gamma_corrected)
        cv.imshow('Image ' + str(gamma), gamma_corrected)

    def pixelVal(pix, r1, s1, r2, s2):
        if (0 <= pix and pix <= r1):
            return (s1 / r1) * pix
        elif (r1 < pix and pix <= r2):
            return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (pix - r2) + s2


    r1 = 70
    s1 = 0
    r2 = 140
    s2 = 255

    pixelVal_vec = np.vectorize(pixelVal)

    contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)
    cv.imshow('Image contrast', contrast_stretched)

    #cv.imwrite('contrast_stretch.jpg', contrast_stretched)

    cv.waitKey(0)
    cv.destroyAllWindows()

def transformaciones_de_colores():
    image = cv.imread('assets/img.png')
    cv.imshow('original', image)

    color1 = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow('COLOR_BGR2HSV', color1)

    #En el color HSV, el canal H solo codifica el color.
    #Los otros dos canales codifican el brillo y saturacion.


    color3 = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    cv.imshow('COLOR_BGR2LAB', color3)
    #En el color LAB, el canal L solo codifica el brillo.
    #Los otros dos canales codifican el color.



    color2 = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    cv.imshow('COLOR_BGR2HLS', color2)
    color6 = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
    cv.imshow('COLOR_BGR2BGRA', color6)
    color4 = cv.cvtColor(image, cv.COLOR_BGR2HLS_FULL)
    cv.imshow('COLOR_BGR2HLS_FULL', color4)
    color5 = cv.cvtColor(image, cv.COLOR_BGR2HSV_FULL)
    cv.imshow('COLOR_BGR2HSV_FULL', color5)
    color7 = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    cv.imshow('COLOR_BGR2Lab', color7)
    color8 = cv.cvtColor(image, cv.COLOR_BGR2LUV)
    cv.imshow('COLOR_BGR2LUV', color8)
    color9 = cv.cvtColor(image, cv.COLOR_BGR2Luv)
    cv.imshow('COLOR_BGR2Luv', color9)
    color10 = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
    cv.imshow('COLOR_BGR2RGBA', color10)
    color11 = cv.cvtColor(image, cv.COLOR_BGR2XYZ)
    cv.imshow('COLOR_BGR2XYZ', color11)
    color12 = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
    cv.imshow('COLOR_BGR2YCR_CB', color12)
    color13 = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    cv.imshow('COLOR_BGR2YCrCb', color13)
    color14 = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow('COLOR_BGR2YUV', color14)
    color15 = cv.cvtColor(image, cv.COLOR_BGR2YUV_I420)
    cv.imshow('COLOR_BGR2YUV_I420', color15)
    color16 = cv.cvtColor(image, cv.COLOR_BGR2YUV_IYUV)
    cv.imshow('COLOR_BGR2YUV_IYUV', color16)
    color17 = cv.cvtColor(image, cv.COLOR_BGR2YUV_YV12)
    cv.imshow('COLOR_BGR2YUV_YV12', color17)

    cv.waitKey(0)
    cv.destroyAllWindows()


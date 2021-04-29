
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def recortar_un_segmento():

    img = cv.imread('assets/img_1.png')
    cv.imshow('Cats', img)
    ax = 100
    ay = 100
    bx = 200
    by = 200
    print(ax, ay, bx, by)
    sug_img = img[ay:by, ax:bx]
    cv.imshow('sug_img', sug_img)


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

def monocromatismo_y_somras():

    par1, par2 = 175, 255

    img = cv.imread('assets/img_1.png')
    cv.imshow('Cats', img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshow, thres = cv.threshold(img,par1, par2, cv.THRESH_BINARY)
    cv.imshow('corte monocromatico', thres)
    threshow, thres = cv.threshold(img, par1, par2, cv.THRESH_BINARY_INV)
    cv.imshow('corte monocromatico inverso', thres)

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

def dibujar_cotornos():

    image = cv.imread('assets/img_7.png')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edged = cv.Canny(gray, 30, 200)
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_NONE
    print(hierarchy)
    cv.imshow('Canny Edges After Contouring 1', edged)
    print("Number of Contours found = " + str(len(contours)))
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv.imshow('Contours 1', image)

    image = cv.imread('assets/img_6.png')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edged = cv.Canny(gray, 30, 200)
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_NONE
    print(hierarchy)
    cv.imshow('Canny Edges After Contouring 2', edged)
    print("Number of Contours found = " + str(len(contours)))
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv.imshow('Contours 2', image)


    image = cv.imread('assets/img_5.png')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edged = cv.Canny(gray, 30, 200)
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_NONE
    print(hierarchy)
    cv.imshow('Canny Edges After Contouring 3', edged)
    print("Number of Contours found = " + str(len(contours)))
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv.imshow('Contours 3', image)

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

    def rotate(img, angle):
        (height, width) = img.shape[:2]
        #punto de rotacion en el centro
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

def rotacion_sin_cortes():

    image = cv.imread('assets/img_5.png')
    '''
    for angle in np.arange(0, 360, 15):
        rotated = imutils.rotate(image, angle)
        cv.imshow("Rotated (Problematic)", rotated)
        cv.waitKey(0)
    '''

    def rotate_bound(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        return cv.warpAffine(image, M, (nW, nH))

    for angle in np.arange(0, 360, 15):
        rotated = rotate_bound(image, angle)
        cv.imshow("Rotated (Correct)", rotated)
        cv.waitKey(0)


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

def leer_imagen():
    img = cv.imread('assets/img.png', 0)
    img_ = cv.cvtColor(img, cv.COLOR_BGR2HSV)



    cv.imshow('5 segundos Cats', img)
    cv.waitKey(5000)
    img_color = cv.imread('assets/img.png')
    cv.imshow('10 segundos color', img_color)
    cv.waitKey(10000)
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

def zoom_imagen():
    import matplotlib.pyplot as plt
    print("nota las dimenciones de la imagen ")
    image = cv.imread('assets/img_1.png ')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    half = cv.resize(image, (0, 0), fx=0.1, fy=0.1)
    bigger = cv.resize(image, (1050, 1610))

    stretch_near = cv.resize(image, (780, 540),
                              interpolation=cv.INTER_NEAREST)

    Titles = ["Original", "Half", "Bigger", "Interpolation Nearest"]
    images = [image, half, bigger, stretch_near]
    count = 4

    for i in range(count):
        plt.subplot(2, 2, i + 1)
        plt.title(Titles[i])
        plt.imshow(images[i])

    plt.show()

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

def extender_imagen():
    image = cv.imread('assets/img_1.png')
    image = cv.resize(image, (340,240))
    cv.imshow('Original Image', image)

    cv.imshow('marco', cv.copyMakeBorder(image, 10, 10, 10, 10, cv.BORDER_CONSTANT))

    cv.imshow('espejo', cv.copyMakeBorder(image, 100, 100, 50, 50, cv.BORDER_REFLECT))

    cv.imshow('mosaico', cv.copyMakeBorder(image, 100, 100, 50, 50, cv.BORDER_WRAP))

    cv.imshow('espejo DEFAULT', cv.copyMakeBorder(image, 100, 100, 50, 50, cv.BORDER_DEFAULT))

    cv.imshow('expandir fondo oscuro', cv.copyMakeBorder(image, 100, 100, 50, 50, cv.BORDER_ISOLATED))

    cv.imshow('correr/replicar ultimo pixel', cv.copyMakeBorder(image, 100, 100, 50, 50, cv.BORDER_REPLICATE))

    cv.waitKey(0)
    cv.destroyAllWindows()

def separar_fondo_de_objeto():
    image = cv.imread('assets/park.jpg')
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # marjen de azul HSV
    lower_blue = np.array([60, 35, 140])
    upper_blue = np.array([180, 255, 255])

    mask = cv.inRange(hsv, lower_blue, upper_blue)

    result = cv.bitwise_and(image, image, mask=mask)

    cv.imshow('original', image)
    cv.imshow('hsv', hsv)
    cv.imshow('mascara', mask)
    cv.imshow('result', result)

    cv.waitKey(0)

    cv.destroyAllWindows()

def caprurar_video():

    captura = cv.VideoCapture(0)
    #video = cv.VideoWriter(video.avi, 0, 1, (width, height))   (* 'MJPG')
    salida = cv.VideoWriter('videoSalida.avi', cv.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
    while (captura.isOpened()):
        ret, imagen = captura.read()
        if ret == True:
            cv.imshow('video', imagen)
            salida.write(imagen)
            if cv.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break
    captura.release()
    salida.release()
    cv.destroyAllWindows()

    #mostrar simplemente video

    '''
    captura = cv.VideoCapture(0)# intentar con 1, 2, 3
    while (captura.isOpened()):
        ret, imagen = captura.read()
        if ret == True:
            cv.imshow('video', imagen)
            if cv.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break
    captura.release()
    cv.destroyAllWindows()
    '''

def caprurar_fotografia():
    # mostrar simplemente foto
    cap = cv.VideoCapture(0)
    leido, frame = cap.read()
    if leido == True:
        cv.imwrite("foto.png", frame)
        print("Foto tomada correctamente")
    else:
        print("Error al acceder a la cámara")
    cap.release()

def borrar_mancha_en_imagen():

    img = cv.imread('assets/park.jpg')
    cv.imshow('Original Image', img)

    blank = np.zeros(img.shape[:2], dtype='uint8')
    mask = cv.rectangle(blank, (325,135),(340, 250), 255, -1)
    cv.imshow('Mascara - blanco en fondo negro', mask)

    dst = cv.inpaint(img, mask, 3, cv.INPAINT_NS)
    cv.imshow('elemento borrado', dst)

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

def intento_de_transformacion_de_perspeectiva():
    img = cv.imread('assets/img_1.png')

    # https://www.geeksforgeeks.org/image-registration-using-opencv-python/
    # https: // opencv - python - tutroals.readthedocs.io / en / latest / py_tutorials / py_imgproc / py_geometric_transformations / py_geometric_transformations.html

    '''
    rows, cols = img.shape
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv.warpAffine(img, M, (cols, rows))

    cv.imshow('img', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''
    '''
    img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)
    height, width = img2.shape

    orb_detector = cv.ORB_create(5000)

    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(d1, d2)

    matches.sort(key=lambda x: x.distance)

    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    homography, mask = cv.findHomography(p1, p2, cv.RANSAC)

    print(type(homography))


    
    transformed_img = cv.warpPerspective(img1_color, homography, (width, height))

    cv.imwrite('output.jpg', transformed_img)
    '''

def intento_de_restar_el_fondo():
    img0 = cv.imread('assets/park.jpg')
    img1 = cv.imread('assets/img.png')
    img2 = cv.imread('assets/img_1.png')
    #BackgroundSubtractorMOG
    #BackgroundSubtractorMOG

    fgmask0 = cv.createBackgroundSubtractorMOG2().apply(img0)
    fgmask1 = cv.createBackgroundSubtractorMOG2().apply(img1)
    fgmask2 = cv.createBackgroundSubtractorMOG2().apply(img2)

    cv.imshow('macara de quitar el fondo 0', fgmask0)
    cv.imshow('macara de quitar el fondo 1', fgmask1)
    cv.imshow('macara de quitar el fondo 2', fgmask2)


    averageValue0 = np.float32(img0)
    cv.accumulateWeighted(img0, averageValue0, 2)
    resultingFrames0 = cv.convertScaleAbs(averageValue0)
    cv.imshow('difuminar el fondo 0', resultingFrames0)

    averageValue1 = np.float32(img1)
    cv.accumulateWeighted(img1, averageValue1, 2)
    resultingFrames1 = cv.convertScaleAbs(averageValue1)
    cv.imshow('difuminar el fondo 1', resultingFrames1)

    averageValue2 = np.float32(img2)
    cv.accumulateWeighted(img2, averageValue2, 2)
    resultingFrames2 = cv.convertScaleAbs(averageValue2)
    cv.imshow('difuminar el fondo 2', resultingFrames2)

    cv.waitKey(0)
    cv.destroyAllWindows()

def restar_el_fondo():

    nivel = 3 #1-5 (o mas)

    image = cv.imread('assets/img_1.png')
    cv.imshow('original', image)

    backgroundModel = np.zeros((1, 65), np.float64)

    foregroundModel = np.zeros((1, 65), np.float64)

    rectangle = (60, 130, 490, 428)

    blank = np.zeros(image.shape[:2], dtype='uint8')
    mask = cv.rectangle(blank, rectangle, 255, -1)
    cv.imshow('Mascara - blanco en fondo negro', mask)

    cv.grabCut(image, mask, rectangle,
                backgroundModel, foregroundModel,
                nivel, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    image = image * mask2[:, :, np.newaxis]

    cv.imshow('recorte', image)

    cv.waitKey(0)
    cv.destroyAllWindows()

def transformaciones_de_colores():
    image = cv.imread('assets/img.png')
    cv.imshow('original', image)

    color1 = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow('COLOR_BGR2HSV', color1)
    color2 = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    cv.imshow('COLOR_BGR2HLS', color2)
    color3 = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    cv.imshow('COLOR_BGR2LAB', color3)
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

def click_en_imagen():

    def click_event(event, x, y, flags, params):

        if event == cv.EVENT_LBUTTONDOWN:
            print(x, ' - ', y, flags, params)

            font = cv.FONT_ITALIC
            cv.putText(img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 1)
            cv.imshow('image', img)

        if event == cv.EVENT_RBUTTONDOWN:
            print(x, ' - ', y)

            font = cv.FONT_ITALIC
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            color = "rgb: " + str(b) + ',' + str(g) + ',' + str(r)
            print(color)
            cv.putText(img, color, (x, y), font, 1, (255, 0, 0), 1)
            cv.imshow('image', img)

    img = cv.imread('assets/img_6.png', 1)

    cv.imshow('image', img)

    cv.setMouseCallback('image', click_event)

    cv.waitKey(0)

    cv.destroyAllWindows()

def ejecutable_desde_consola_y_argumentos_al_script():

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the input image")
    ap.add_argument("-m", "--method", required=True, help="Sorting method")
    args = vars(ap.parse_args())

    image = cv.imread(args["image"])
    cv.imshow('original', image)

    print(args["method"])

    '''
    $ python name_script.py --image images/image.png --method "right-to-left"
    '''

import cv2 as cv
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

def recortar_imagen_con_mascara():
    img = cv.imread('assets/img_1.png')
    cv.imshow('Cats', img)
    blank = np.zeros(img.shape[:2], dtype='uint8')
    mask = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)
    masked = cv.bitwise_and(img, img, mask=mask)
    cv.imshow('Mask', masked)

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

def contornos_internos_y_externos():

    image = cv.imread('assets/img_7.png')
    imageRETR_LIST = image.copy()
    imageRETR_EXTERNAL = image.copy()
    imageRETR_TREE = image.copy()
    imageRETR_CCOMP = image.copy()
    imageRETR_FLOODFILL = image.copy()

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edged = cv.Canny(gray, 30, 200)
    #cv.imshow('edged', edged)

    #contours, hierarchy = cv.findContours(edged, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_NONE
    #print(hierarchy)
    #print("cv.RETR_LIST - " + str(len(contours)))
    #cv.drawContours(imageRETR_LIST, contours, -1, (0, 255, 0), 2)
    #cv.imshow('RETR_LIST', imageRETR_LIST)

    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_NONE
    print("cv.RETR_EXTERNAL - " + str(len(contours)))
    print(hierarchy)
    #cv.drawContours(imageRETR_EXTERNAL, contours, -1, (0, 255, 0), 2)
    #cv.imshow('RETR_EXTERNAL', imageRETR_EXTERNAL)

    #contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_NONE
    #print("cv.RETR_TREE - " + str(len(contours)))
    #print(hierarchy)
    #cv.drawContours(imageRETR_TREE, contours, -1, (0, 255, 0), 2)
    #cv.imshow('RETR_TREE', imageRETR_TREE)

    #contours, hierarchy = cv.findContours(edged, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_NONE
    #print("cv.RETR_CCOMP - " + str(len(contours)))
    #print(hierarchy)
    #cv.drawContours(imageRETR_CCOMP, contours, -1, (0, 255, 0), 2)
    #cv.imshow('RETR_CCOMP', imageRETR_CCOMP)


    for c in contours:
        if cv.contourArea(c) < 100:
            continue
        cv.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv.imshow('image', image)
        cv.waitKey(0)

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

def transformacion_de_perspectiva():
    imagen = cv.imread('assets/img_3.png')
    ax, ay = 100, 50; bx, by = 400, 100
    cx, cy = 50, 350; dx, dy = 350, 400

    cv.circle(imagen, (ax, ay), 2, (255, 0, 0), 2)
    cv.circle(imagen, (bx, by), 2, (0, 255, 0), 2)
    cv.circle(imagen, (cx, cy), 2, (0, 0, 255), 2)
    cv.circle(imagen, (dx, dy), 2, (255, 255, 0), 2)

    mx, my = 400, 400

    pts1 = np.float32([[ax, ay], [bx, by], [cx, cy], [dx, dy]])
    pts2 = np.float32([[0, 0], [mx, 0], [0, my], [mx, my]])

    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(imagen, M, (mx, my))  # dimencion absoluta

    cv.imshow('Imagen', imagen)
    cv.imshow('dst', dst)
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

def separar_fondo_de_cierto_tono_con_mascara():
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

def quitado_de_fondo_inteligente():

    nivel = 3 #1-5 (o mas)

    image = cv.imread('assets/img_1.png')
    cv.imshow('original', image)

    backgroundModel = np.zeros((1, 65), np.float64)

    foregroundModel = np.zeros((1, 65), np.float64)

    rectangle = (60, 130, 490, 428)

    blank = np.zeros(image.shape[:2], dtype='uint8')
    mask = cv.rectangle(blank, rectangle, 255, -1)
    cv.imshow('Mascara - blanco en fondo negro', mask)
    print("recuerda que la mascara no tiene que ser cuadrada ")
    cv.grabCut(image, mask, rectangle,
                backgroundModel, foregroundModel,
                nivel, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    image = image * mask2[:, :, np.newaxis]

    cv.imshow('recorte', image)

    cv.waitKey(0)
    cv.destroyAllWindows()

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

def simetrias():

    image = cv.imread("assets/img_14.png")
    cv.imshow('image', image)
    flip_1 = cv.flip(image, 0)
    cv.imshow('flip 0', flip_1)
    flip_2 = cv.flip(image, 1)
    cv.imshow('flip 1', flip_2)
    flip_3 = cv.flip(image, -1)
    cv.imshow('flip -1', flip_3)

    cv.waitKey(0)
    cv.destroyAllWindows()

def operacion_numerica_de_imagenes():

    img1 = cv.imread('assets/img_1.png', )
    img1 = cv.resize(img1, (400, 300), interpolation=cv.INTER_CUBIC)

    img2 = cv.imread('assets/img_2.png', )
    img2 = cv.resize(img2, (400, 300), interpolation=cv.INTER_CUBIC)

    add = cv.add(img2, img1)
    cv.imshow('add', add)

    sub = cv.subtract(img2, img1)
    cv.imshow('sub', sub)

    abs = cv.absdiff(img2, img1)
    cv.imshow('abs', abs)

    cv.waitKey(0)
    cv.destroyAllWindows()


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import imutils
from imutils import contours


def seleccion_de_formas_por_dilatacion_y_contraccion():
    img = cv.imread('assets/img_1.png')
    dimensions = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
    img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    cv.imshow('original', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #-----

    ret, thresh = cv.threshold(gray, 0, 255,
                                cv.THRESH_BINARY_INV +
                                cv.THRESH_OTSU)
    cv.imshow('image thresh invertida', thresh)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

    #dilatado = cv.dilate(thres.copy(), None, iterations=1)
    #dilatado = cv.erode(thres.copy(), None, iterations=1)

    bg = cv.dilate(closing, kernel, iterations=1)

    cv.imshow('image dilatada invertida', bg)

    dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 0)
    ret, fg = cv.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)

    cv.imshow('image contraida invertida', fg)

    #-----

    ret, thresh = cv.threshold(gray, 0, 255,
                                cv.THRESH_OTSU)
    cv.imshow('image thresh', thresh)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

    bg = cv.dilate(closing, kernel, iterations=1)
    cv.imshow('image dilatada', bg)

    dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 0)
    ret, fg = cv.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)

    cv.imshow('image contraida', fg)

    cv.waitKey(0)
    cv.destroyAllWindows()

def gradiente_de_bordes_por_colores():
    image = cv.imread('assets/img.png')
    cv.imshow('original', image)

    color1 = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow('COLOR_BGR2HSV', color1)

    blue1 = np.array([50, 10, 10])#110,50,50
    blue2 = np.array([130, 255, 255])

    mask = cv.inRange(color1, blue1, blue2)

    res = cv.bitwise_and(image, image, mask=mask)

    kernel = np.ones((5, 5), np.uint8)

    gradient = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernel)

    cv.imshow('Gradient', gradient)

    cv.waitKey(0)
    cv.destroyAllWindows()

def quitar_ruido_suabizar_y_ver_bordes():
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
    #dilatado = cv.dilate(thres.copy(), None, iterations=1)
    #img = cv.dilate(img, (7, 7), iterations=5)
    #img = cv.erode(img, (7, 7), iterations=3)

    bilateral = cv.bilateralFilter(img, 15, 75, 75)
    cv.imshow('imagen con color osfet', bilateral)

    img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    cv.imshow('imagen suabizada', img)

    cv.imshow('canny bordes principales', cv.Canny(img, 200,255))
    cv.imshow('canny todos bordes', cv.Canny(img, 0, 50))
    cv.waitKey(0)
    cv.destroyAllWindows()

def detectar_lineas():

    img = cv.imread('assets/img_6.png')
    #dimensions = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
    #img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY )

    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    cv.imshow('edges', edges)

    ret, thresh = cv.threshold(edges, 0, 255,
                                cv.THRESH_BINARY +
                                cv.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

    #dilatado = cv.dilate(thres.copy(), None, iterations=1)
    bg = cv.dilate(closing, kernel, iterations=1)
    cv.imshow('bg', bg)

    largo_minimo = 10
    maximo_brecha_de_linea = 100
    lines = cv.HoughLinesP(bg, 1, np.pi / 180, 100, largo_minimo, maximo_brecha_de_linea)
    print(lines)
    for a in range(len(lines)):
        for x1, y1, x2, y2 in lines[a]:
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('img', img)

    cv.waitKey(0)
    cv.destroyAllWindows()

def detectar_circulos():

    img = cv.imread("assets/img_4.png")
    cv.imshow("original", img)

    gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gauss = cv.GaussianBlur(gris, (3, 3), 0)

    cv.imshow("gauss", gauss)
    #https://www.geeksforgeeks.org/circle-detection-using-opencv-python/
    detected_circles = cv.HoughCircles(gauss,
                                        cv.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=40, minRadius=30, maxRadius=50)

    print(detected_circles)
    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv.circle(img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv.circle(img, (a, b), 1, (0, 0, 255), 3)

    cv.imshow("Detected Circle", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def deteccion_de_vertices():
    img1 = cv.imread("assets/img_5.png")
    #cv.imshow("original", img1)

    n_esquinas_a_buscar = 5

    gris = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gauss = cv.GaussianBlur(gris, (3, 3), 0)
    corners = cv.goodFeaturesToTrack(gauss, n_esquinas_a_buscar, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv.circle(img1, (x, y), 3, 255, -1)

    cv.imshow("img1", img1)


    img2 = cv.imread("assets/img_6.png")
    #cv.imshow("original", img2)

    n_esquinas_a_buscar = 17

    gris = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    gauss = cv.GaussianBlur(gris, (3, 3), 0)
    corners = cv.goodFeaturesToTrack(gauss, n_esquinas_a_buscar, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv.circle(img2, (x, y), 3, 255, -1)

    cv.imshow("img2", img2)


    cv.waitKey(0)
    cv.destroyAllWindows()

def deteccion_de_esquinas_no_se_entinde():
    image = cv.imread("assets/img_7.png")
    # cv.imshow("original", img1)
    operatedImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    operatedImage = np.float32(operatedImage)

    dest = cv.cornerHarris(operatedImage, 5, 5, 0.1)

    print(dest)

    #dilatado = cv.dilate(thres.copy(), None, iterations=1)
    dest = cv.dilate(dest, None)

    for a in range(len(dest)):
        if(np.sum(dest[a])> 0.1):
            print(dest[a])

    image[dest > 0.01 * dest.max()] = [0, 0, 255]

    cv.imshow('Image with Borders', image)

    cv.waitKey(0)
    cv.destroyAllWindows()

def deteccion_de_esquinas():

    img = cv.imread("assets/img_7.png")

    n_esquinas_a_buscar = 5

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray_img, n_esquinas_a_buscar, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 3, (255, 0, 0), -1)


    cv.imshow('Image with Borders', img)

    cv.waitKey(0)
    cv.destroyAllWindows()

def deteccion_de_circulos_no_se_entinde():

    image = cv.imread("assets/img_9.png", 0)
    ret, image = cv.threshold(image, 0, 255,
                               cv.THRESH_BINARY +
                               cv.THRESH_OTSU)
    cv.imshow("image", image)

    params = cv.SimpleBlobDetector_Params()

    params.filterByArea = True       #puede desactivar los parametros que no use
    params.minArea = 10
    params.filterByCircularity = True
    params.minCircularity = 0.8       #calidad del circulo 1= circulo perfecto (en la practica ninguno)
    params.filterByConvexity = False
    params.minConvexity = 0.5
    params.filterByInertia = False
    params.minInertiaRatio = 0.1

    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    list_key = cv.KeyPoint_convert(keypoints)[1]
    print(list_key)

    blank = np.zeros((1, 1))
    blobs = cv.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                              cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    print(text)
    cv.putText(blobs, text, (20, 550),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv.circle(blobs, (420, 634), 5, (0, 255, 0), 2)
    cv.circle(blobs, (666, 629), 5, (0, 255, 0), 2)
    cv.imshow("Filtering Circular Blobs Only", blobs)
    cv.waitKey(0)
    cv.destroyAllWindows()

def coincidencia_de_imagen_dentro_de_otra():

    field_threshold = {"Mexico": 0.8,#.7 rigurocidad de coincidencia
                       "Espanol": 0.8,#.6 rigurocidad de coincidencia
                       }
    # Function to Generate bounding
    # boxes around detected fields
    def getBoxed(img, img_gray, template, field_name="Mexico"):
        w, h = template.shape[::-1]

        res = cv.matchTemplate(img_gray, template,
                                cv.TM_CCOEFF_NORMED)

        hits = np.where(res >= field_threshold[field_name])

        for pt in zip(*hits[::-1]):
            print("se encontro una coincidencia")
            cv.rectangle(img, pt, (pt[0] + w, pt[1] + h),
                          (0, 255, 255), 2)

            y = pt[1] - 10 if pt[1] - 10 > 10 else pt[1] + h + 20

            cv.putText(img, field_name, (pt[0], y),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        return img

    img = cv.imread('assets/img_10.png')

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    template_esp = cv.imread('assets/esp.png', 0)
    template_mex = cv.imread('assets/mex.png', 0)

    img = getBoxed(img.copy(), img_gray.copy(),
                   template_esp, 'Espanol')

    img = getBoxed(img.copy(), img_gray.copy(),
                   template_mex, 'Mexico')

    cv.imshow('Detected', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def encontrar_y_borrar_puntos_pequenos():

    img = cv.imread('assets/img_12.png')
    #dimensions = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
    #img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY )

    edges = cv.Canny(gray, 200, 250, apertureSize=3)
    cv.imshow('edges', edges)

    ret, thresh = cv.threshold(edges, 100, 255,
                                cv.THRESH_BINARY +
                                cv.THRESH_OTSU)

    cnts = cv.findContours(thresh, cv.RETR_LIST,
                            cv.CHAIN_APPROX_SIMPLE)[-2]

    s1 = 1
    s2 = 30
    xcnts = []
    for cnt in cnts:
        if s1 < cv.contourArea(cnt) < s2:
            xcnts.append(cnt)
    memoria = 0
    for a in xcnts:
        for b in a[0]:
            memoria += 1
            cv.circle(img, (b[0],b[1]), 10, (0, 255, 255), -1)
            cv.circle(gray, (b[0],b[1]), 10, (0), -1)

    #for a1, b2 in xcnts:
    #    print(a1,b2)

    cv.imshow('gray', gray)
    cv.imshow('img', img)
    print("memoria", memoria)
    print(len(xcnts))

    cv.waitKey(0)
    cv.destroyAllWindows()

def slider_tienpo_real():# se podria hacer uno que sea manual

    def hcer_algo():
        print("hcer_algo")

    image = np.zeros((512, 512, 3), np.uint8)
    windowName = "Open CV Color Palette"

    cv.namedWindow(windowName)

    cv.createTrackbar('Blue', windowName, 0, 255, hcer_algo)
    cv.createTrackbar('Green', windowName, 0, 255, hcer_algo)
    cv.createTrackbar('Red', windowName, 0, 255, hcer_algo)

    while (True):
        cv.imshow(windowName, image)

        if cv.waitKey(1) == 27:
            break

        blue = cv.getTrackbarPos('Blue', windowName)
        green = cv.getTrackbarPos('Green', windowName)
        red = cv.getTrackbarPos('Red', windowName)

        image[:] = [blue, green, red]
        print(blue, green, red)

    cv.destroyAllWindows()

def cordenadas_de_esquinas():
    font = cv.FONT_HERSHEY_COMPLEX

    img2 = cv.imread('assets/img_7.png', cv.IMREAD_COLOR)
    cv.imshow('img2', img2)
    img = cv.imread('assets/img_7.png', cv.IMREAD_GRAYSCALE)
    cv.imshow('img', img)

    _, threshold = cv.threshold(img, 110, 255, cv.THRESH_BINARY)
    cv.imshow('threshold', threshold)

    contours, _ = cv.findContours(threshold, cv.RETR_TREE,
                                   cv.CHAIN_APPROX_SIMPLE)

    #print(contours)

    for a in range(len(contours)):
        for b in contours[a]:
            print(b)
        print("-")

    for cnt in contours:

        approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True)

        cv.drawContours(img2, [approx], 0, (0, 0, 255), 5)

        n = approx.ravel()
        i = 0

        for j in n:
            if (i % 2 == 0):
                x = n[i]
                y = n[i + 1]

                string = str(x) + " " + str(y)

                if (i == 0):
                    cv.putText(img2, "new " + string, (x, y),
                                font, 0.4, (0, 255, 0))
                else:
                    cv.putText(img2, string, (x, y),
                                font, 0.4, (0, 255, 0))
            i = i + 1

    cv.imshow('image2', img2)

    cv.waitKey(0)
    cv.destroyAllWindows()

def rec_de_objetos_por_threshold_y_refinamineto():


    par1, par2 = 220, 255

    img = cv.imread('assets/img_4.png')
    #cv.imshow('img', img)


    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    t, thres_g = cv.threshold(gray, par1, par2, cv.THRESH_BINARY)
    cv.imshow('thres1', thres_g)

    erode = cv.erode(thres_g.copy(), None, iterations=2)
    cv.imshow('erode refinamineto', erode)

    dilatado = cv.dilate(erode.copy(), None, iterations=2)
    cv.imshow('dilatado refinamineto', dilatado)

    thres_g = cv.threshold(dilatado, 225, 255, cv.THRESH_BINARY_INV)[1]
    cv.imshow("thres2", thres_g)

    mask = thres_g.copy()
    sin_fondo = cv.bitwise_and(img, img, mask=mask)
    cv.imshow("sin_fondo", sin_fondo)

    cnts = cv.findContours(thres_g.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    output = img.copy()
    print(len(cnts))

    cv.drawContours(output, cnts, -1, (240, 0, 159), 3)

    text = "I found {} objects!".format(len(cnts))
    cv.putText(output, text, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2)
    cv.imshow("Contours", output)

    cv.waitKey(0)
    cv.destroyAllWindows()

def rec_de_objetos_por_canny_y_contraste():
    # Cargamos la imagen
    img = cv.imread("assets/img_4.png")
    cv.imshow("original", img)

    selct = 1

    if selct == 1:
        img = cv.GaussianBlur(img, (7, 7), 0)
        img = cv.medianBlur(img, 5)
        img = cv.medianBlur(img, 5)
    elif selct == 2: img = cv.medianBlur(img, 5)
    elif selct == 3: img = cv.bilateralFilter(img, 9, 75, 75)
    cv.imshow("suavizado", img)

    #                                          [0.5, 1.5, 2.0]
    Image_gamma = np.array(255 * (img / 255) ** 2.5, dtype='uint8')
    cv.imshow('Image_gamma', Image_gamma)

    gris = cv.cvtColor(Image_gamma, cv.COLOR_BGR2GRAY)

    canny = cv.Canny(gris, 200, 255)
    cv.imshow("canny", canny)

    (contornos, _) = cv.findContours(canny.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("He encontrado {} objetos".format(len(contornos)))

    cv.drawContours(img, contornos, -1, (0, 0, 255), 2)
    cv.imshow("contornos", img)

    cv.waitKey(0)
    cv.destroyAllWindows()

def rec_de_objetos_por_color():

    image = cv.imread("assets/img.png")
    cv.imshow("original", image)

    '''
    image_copy = image.copy()
    for a in range(len(image_copy)):
        for b in range(len(image_copy[0])):
            if np.array_equal(image_copy[a][b], np.array([255, 82, 140])):
                image_copy[a][b] = np.array([89, 222, 255])
    cv.imshow("pinada busqueda de color (fuerza bruta)", image_copy)
    '''

    lower = np.array([245, 70, 130])#bgr 255 82 140
    upper = np.array([255, 90, 150])
    shapeMask = cv.inRange(image, lower, upper)

    pielb = np.full(image.shape[:2], (89), dtype='uint8')
    pielg = np.full(image.shape[:2], (222), dtype='uint8')
    pielr = np.full(image.shape[:2], (255), dtype='uint8')
    # amarillo rgb(255, 222, 89)

    color_amarillo = cv.merge([pielb, pielg, pielr])

    mask = shapeMask.copy()
    mascara_de_color = cv.bitwise_and(color_amarillo, color_amarillo, mask=mask)
    cv.imshow("mascara_de_color", mascara_de_color)

    for a in range(len(mascara_de_color)):
        for b in range(len(mascara_de_color[0])):
            for c in range(3):
                if mascara_de_color[a][b][c] != 0:
                    image[a][b][c] = mascara_de_color[a][b][c]

    cv.imshow('pinada selectivamente con masara (fuerza bruta)', image)

    cnts = cv.findContours(shapeMask.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("I found {} black shapes".format(len(cnts)))
    #cv.imshow("Mask", shapeMask)
    cv.drawContours(image, cnts, -1, (0, 255, 0), 2)

    #cv.imshow("Image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def rec_de_objetos_por_orientaion_y_area():

    image = cv.imread("assets/img_15.png")
    cv.imshow("original", image)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)

    edged = cv.Canny(gray, 50, 100)
    edged = cv.dilate(edged, None, iterations=1)
    edged = cv.erode(edged, None, iterations=1)
    cv.imshow("edged", edged)

    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #oden de derecha a isquierda
    (cnts, _) = contours.sort_contours(cnts)

    for (i, c) in enumerate(cnts):
        if cv.contourArea(c) < 100:
            continue

        box = cv.minAreaRect(c)
        box = cv.boxPoints(box)
        box = np.array(box, dtype="int")
        cv.drawContours(image, [box], -1, (0, 255, 0), 2)

        print("Object #{}:".format(i + 1))
        print(box.astype("int"))
        #print(box)
        print("")

        colors = ((0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 0, 0))

        for ((x, y), color) in zip(box, colors):
            cv.circle(image, (int(x), int(y)), 5, color, -1)

        cv.putText(image, "#{}".format(i + 1),
                    (int(box[0][0] - 15), int(box[0][1] - 15)),
                    cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv.imshow("Image", image)

    cv.waitKey(0)
    cv.destroyAllWindows()

def rec_de_objeto_mas_grande_y_cordenadas_de_exremos():
    image = cv.imread("assets/img_16.png")
    #cv.imshow("original", image)
    gris = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gris = cv.GaussianBlur(gris, (7, 7), 0)

    thresh = cv.threshold(gris, 45, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)
    #cv.imshow("thresh", thresh)


    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #seleionar objeto mas grande
    c = max(cnts, key=cv.contourArea)
    print(cv.moments(c))

    #cordenadas de exremos carttecianos
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    cv.drawContours(image, [c], -1, (0, 255, 255), 2)
    #cv.imshow("drawContours", image)

    cv.circle(image, extLeft, 8, (0, 0, 255), -1)
    cv.circle(image, extRight, 8, (0, 255, 0), -1)
    cv.circle(image, extTop, 8, (255, 0, 0), -1)
    cv.circle(image, extBot, 8, (255, 255, 0), -1)

    cv.imshow("Image", image)

    cv.waitKey(0)
    cv.destroyAllWindows()

#c = max(cnts, key=cv.isContourConvex)
#https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
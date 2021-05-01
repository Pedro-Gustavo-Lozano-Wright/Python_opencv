
import cv2 as cv
import numpy as np


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

def ajustar_parametros_y_enviar_datos_mientras_corre_video():

    cap = cv.VideoCapture('assets/personas.mp4')
    while True:
        ret, frame = cap.read()
        if ret == False: break

        # frame = imutils.resize(frame, width=640)
        # frame = cv.flip(frame, 1)
        cv.imshow('Frame', frame)
        k = cv.waitKey(20)#ajustar
        if k == ord('a'):
            print("operacion mienras corre video")
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()

def ejecutable_desde_consola_y_argumentos_al_script():

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the input image")
    ap.add_argument("-m", "--method", required=True, help="Sorting method")
    args = vars(ap.parse_args())

    #import sys
    #image_sys = cv.imread(sys.argv[1])
    #cv.imshow('image_sys', image_sys)

    image = cv.imread(args["image"])
    cv.imshow('image', image)

    print(args["method"])

    '''
    $ python name_script.py --image images/image.png --method "right-to-left"
    '''


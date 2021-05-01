#from scipy.spatial import distance as dist
#from imutils import perspective
import pytesseract as pytesseract
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2 as cv
import math

def distancias_relativas():


    image = cv.imread("assets/img_17.png")
    ref = cv.imread("assets/img_3.png")

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)

    edged = cv.Canny(gray, 50, 100)
    edged = cv.dilate(edged, None, iterations=1)
    edged = cv.erode(edged, None, iterations=1)

    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    for c in cnts:

        if cv.contourArea(c) < 100:
            continue

        #print("area ", cv.contourArea(c))
        orig = image.copy()
        box = cv.minAreaRect(c)
        #print("area Rect", cv.minAreaRect(c))
        box = cv.boxPoints(box)# box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
        #print("(pocicion) y (largo y ancho)", cv.boxPoints(box))
        box = np.array(box, dtype="int")

        cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        # loop over the original points and draw them
        for (x, y) in box:
            cv.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = ((tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5)
        (blbrX, blbrY) = ((bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5)

        (tlblX, tlblY) = ((tl[0] + bl[0]) * 0.5, (tl[1] + bl[1]) * 0.5)
        (trbrX, trbrY) = ((tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5)
        # draw the midpoints on the image
        cv.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        # draw lines between the midpoints
        cv.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)
        cv.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        print("dA", (tltrX, tltrY), (blbrX, blbrY))
        print("dB", (tlblX, tlblY), (trbrX, trbrY))


        dA = ((blbrY - tltrY) * (blbrY - tltrY))
        cA = ((blbrX - tltrX) * (blbrX - tltrX))
        dA = math.sqrt(dA + cA)


        dB = (trbrX - tlblX) * (trbrX - tlblX)
        cB = (trbrY - tlblY) * (trbrY - tlblY)
        dB = math.sqrt(dB + cB)


        cv.putText(orig, "{:.1f}".format(dA),#dimA
                    (int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv.putText(orig, "{:.1f}".format(dB),#dimB
                    (int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        #cv.imshow("ref", ref)

        cv.imshow("Image", orig)

        cv.waitKey(0)
        cv.destroyAllWindows()

def centre_de_objeto():

    image = cv.imread("assets/img_17.png")

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_TOZERO_INV)[1]
    #                               THRESH_TOZERO_INV THRESH_TOZERO
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]
    #                               THRESH_TOZERO_INV THRESH_TOZERO

    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv.contourArea(c) < 100:
            continue

        M = cv.moments(c)# encontrar el centro
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv.putText(image, "center", (cX - 20, cY - 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv.imshow("Image", image)
        cv.waitKey(0)

    cv.waitKey(0)
    cv.destroyAllWindows()

def orden_carteciano_de_objetos():
    #liguero margen de error de 5 px
    def sort_contours(cnts, method="left-to-right"):

        reverse = False
        i = 0

        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        boundingBoxes = [cv.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        return (cnts, boundingBoxes)

    def draw_contour(image, c, i):

        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)

        return image

    image = cv.imread("assets/img_15.png")

    accumEdged = np.zeros(image.shape[:2], dtype="uint8")

    for chan in cv.split(image):

        chan = cv.medianBlur(chan, 11)
        edged = cv.Canny(chan, 50, 200)
        accumEdged = cv.bitwise_or(accumEdged, edged)

    cv.imshow("Edge Map", accumEdged)

    cnts = cv.findContours(accumEdged.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]
    '''
    orig = image.copy()
    for (i, c) in enumerate(cnts):
        orig = draw_contour(orig, c, i)
    cv.imshow("desorden", orig)
    '''

    metodos_orden = ["left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"]

    for metod in metodos_orden:
        orig_orden = image.copy()
        (cnts_copy, boundingBoxes) = sort_contours(cnts, metod)
        for (i, c) in enumerate(cnts_copy):

            M = cv.moments(c)  # encontrar el area
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv.circle(image, (cX, cY), 2, (255, 255, 255), -1)
            cv.drawContours(image, [c], -1, (0, 255, 0), 2)

            draw_contour(orig_orden, c, i)
        cv.imshow(str(metod), orig_orden)



    cv.waitKey(0)
    cv.destroyAllWindows()

def deteccon_geometrica_de_objeto():

    image = cv.imread("assets/img_15.png")

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]
    #                               THRESH_TOZERO_INV THRESH_TOZERO

    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #                               THRESH_TOZERO_INV THRESH_TOZERO

    for c in cnts:

        if cv.contourArea(c) < 100:
            continue

        M = cv.moments(c)
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))

        shape = "unidentified"
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 3:
            shape = "triangulo"
        elif len(approx) == 4:
            (x, y, w, h) = cv.boundingRect(approx)
            ar = w / float(h)
            shape = "cuadrado" if ar >= 0.95 and ar <= 1.05 else "rectangulo"
        elif len(approx) == 5:
            shape = "pentagono"
        elif len(approx) == 6:
            shape = "exagono"
        elif len(approx) == 7:
            shape = "septagono"
        else: shape = "circle"

        global anx
        anx = 0
        global any
        any = 0
        global er_poit
        global ulr_poit
        er_poit = approx[0][0]
        ulr_poit = approx[-1][0]
        for xypoint in approx:
            for ax, ay in xypoint:
                print(xypoint[0])
                print(ax, ay)
                if (er_poit[0] == ax) & (er_poit[1] == ay):
                    anx = ax
                    any = ay
                    continue
                cv.line(thresh, (ax, ay), (anx,any) , (0,0,0),2)
                if (ulr_poit[0] == ax) & (ulr_poit[1] == ay):
                    cv.line(thresh, (ax, ay), (er_poit[0], er_poit[1]), (0, 0, 0), 2)
                anx = ax
                any = ay

        c = c.astype("float")
        c = c.astype("int")
        cv.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv.putText(image, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

    cv.imshow("Image", image)
    cv.imshow("thresh", thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()

def encontrar_y_leer_7_segmentos():
    from imutils import contours
    import imutils

    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0

        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        boundingBoxes = [cv.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        return cnts, boundingBoxes


    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 1, 0): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }

    image = cv.imread("assets/img_19.png")
    cv.imshow("Image", image)
    cv.waitKey(0)

    #image = imutils.resize(image, height=500)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    edged = cv.Canny(blurred, 50, 200, 255)
    cv.imshow("edged", edged)
    cv.waitKey(0)
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    global sub_image

    for c in cnts:
        if cv.contourArea(c) < 1000:
            continue

        box = cv.minAreaRect(c)
        # print("area Rect", cv.minAreaRect(c))
        box = cv.boxPoints(box)  # box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
        # print("(pocicion) y (largo y ancho)", cv.boxPoints(box))
        box = np.array(box, dtype="int")

        global ax
        global ay
        ax = (box[1][0])
        ay = (box[1][1])
        bx = (box[3][0])
        by = (box[3][1])
        #print(ax,ay,bx,by)
        sub_image = image[ay+5:by-5, ax+5:bx-5]
        cv.imshow("sug_image", sub_image)
        cv.waitKey(0)

        for (x, y) in box:
            print("x:", x,  "y:", y)
            cv.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        cv.drawContours(image, [c], -1, (0, 255, 0), 2)

        cv.waitKey(0)

    sub_gray = cv.cvtColor(sub_image, cv.COLOR_BGR2GRAY)

    thresh = cv.threshold(sub_gray, 0, 255,
                           cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    #                               THRESH_TOZERO_INV THRESH_TOZERO

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 3))
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    #                               THRESH_TOZERO_INV THRESH_TOZERO

    cv.imshow("thresh", thresh)
    cv.waitKey(0)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)



    for c in cnts:
        if cv.contourArea(c) < 100:
            continue

        cv.drawContours(sub_image, [c], -1, (0, 255, 0), 2)

        cv.imshow("sub_image", sub_image)
        cv.waitKey(0)


    digitCnts = []


    for c in cnts:

        (x, y, w, h) = cv.boundingRect(c)

        if w >= 15 and (h >= 30 and h <= 40):
            digitCnts.append(c)

    digitCnts = sort_contours(digitCnts)[0]
    digits = []

    print("digitCnts", len(digitCnts))

    for c in digitCnts:

        (x, y, w, h) = cv.boundingRect(c)


        roi = thresh[y:y + h, x:x + w]

        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)

        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)),  # top-left
            ((w - dW, 0), (w, h // 2)),  # top-right
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),  # bottom-left
            ((w - dW, h // 2), (w, h)),  # bottom-right
            ((0, h - dH), (w, h))  # bottom
        ]
        on = [0] * len(segments)

        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):

            segROI = roi[yA:yB, xA:xB]
            total = cv.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)

            if total / float(area) > 0.5:
                on[i] = 1
        print(tuple(on))

        try:
            digit = DIGITS_LOOKUP[tuple(on)]
            digits.append(digit)
            cv.rectangle(sub_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv.putText(sub_image, str(digit), (x - 10, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        except:
            print("An exception occurred")
    

    print(digits)
    cv.imshow("Input", image)

    #cv.imshow("Output", output)
    cv.waitKey(0)

def traking_objeto_en_video():
    import sys

    PY3 = sys.version_info[0] == 3

    if PY3:
        xrange = range

    import numpy as np
    import cv2 as cv
    import asyncio

    async def sleep(time=.1):
        await asyncio.sleep(time)


    class App(object):
        def __init__(self, video_src):
            self.cam = cv.VideoCapture(video_src)
            _ret, self.frame = self.cam.read()
            cv.namedWindow('camshift')
            cv.setMouseCallback('camshift', self.onmouse)

            self.selection = None
            self.drag_start = None
            self.show_backproj = False
            self.track_window = None

        def onmouse(self, event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                self.drag_start = (x, y)
                self.track_window = None
            if self.drag_start:
                xmin = min(x, self.drag_start[0])
                ymin = min(y, self.drag_start[1])
                xmax = max(x, self.drag_start[0])
                ymax = max(y, self.drag_start[1])
                self.selection = (xmin, ymin, xmax, ymax)
            if event == cv.EVENT_LBUTTONUP:
                self.drag_start = None
                self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

        def show_hist(self):
            bin_count = self.hist.shape[0]
            bin_w = 24
            img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
            for i in xrange(bin_count):
                h = int(self.hist[i])
                cv.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                             (int(180.0 * i / bin_count), 255, 255), -1)
            img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            cv.imshow('hist', img)

        def run(self):
            while True:

                asyncio.run(sleep())

                _ret, self.frame = self.cam.read()
                vis = self.frame.copy()
                hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
                mask = cv.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

                if self.selection:
                    x0, y0, x1, y1 = self.selection
                    hsv_roi = hsv[y0:y1, x0:x1]
                    mask_roi = mask[y0:y1, x0:x1]
                    hist = cv.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
                    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
                    self.hist = hist.reshape(-1)
                    self.show_hist()

                    vis_roi = vis[y0:y1, x0:x1]
                    cv.bitwise_not(vis_roi, vis_roi)
                    vis[mask == 0] = 0

                if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                    self.selection = None
                    prob = cv.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                    prob &= mask
                    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
                    track_box, self.track_window = cv.CamShift(prob, self.track_window, term_crit)

                    if self.show_backproj:
                        vis[:] = prob[..., np.newaxis]
                    try:
                        cv.ellipse(vis, track_box, (0, 0, 255), 2)
                    except:
                        print(track_box)

                cv.imshow('camshift', vis)

                ch = cv.waitKey(5)
                if ch == 27:
                    break
                if ch == ord('b'):
                    self.show_backproj = not self.show_backproj
            cv.destroyAllWindows()

    try:
        video_src = "assets/personas.mp4"#lago.mp4
    except:
        video_src = 0
    App(video_src).run()

def cosas():
    img1 = cv.imread('assets/img_1.png',)
    img1 = cv.resize(img1, (400, 300), interpolation=cv.INTER_CUBIC)

    img2 = cv.imread('assets/img_2.png',)
    img2 = cv.resize(img2, (400, 300), interpolation=cv.INTER_CUBIC)

    add = cv.add(img2, img1)
    cv.imshow('add', add)

    sub = cv.subtract(img2, img1)
    cv.imshow('sub', sub)

    abs = cv.absdiff(img2, img1)
    cv.imshow('abs', abs)

    cv.waitKey(0)
    cv.destroyAllWindows()


import cv2
import matplotlib.pyplot as plt
import numpy as np


def process(frame):
    '''
    Takes an image in GRAY and apply filters
    '''
    frame = cv2.equalizeHist(frame)
    #frame = cv2.bilateralFilter(frame, 9, 75, 75)
    frame = cv2.GaussianBlur(frame, (5,5), 0.5)
    #frame = cv2.medianBlur(frame, 5)
    return frame
    
def hog_compute(frame, locations):
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist = hog.compute(frame,winStride,padding,locations)
    #1764
    print(max(hist))
    return hist


if __name__ == "__main__":
    video = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./utils/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./utils/haarcascade_eye.xml')
    stop_flag = False
    while True:
        read_flag = False
        read_flag, frame = video.read()
        if not read_flag:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = process(frame)
        # Detect face and get ROI
        hist = []
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        if len(faces) > 0:
            for (x,y,w,h) in faces:
                eyes = eye_cascade.detectMultiScale(frame[y:y+h, x:x+w])
                if len(eyes) == 2:
                    frame = frame[y:y+h, x:x+w]
                    hist = hog_compute(frame, (w/2, h/2))
                    fig = plt.figure()
                    ax1 = fig.add_subplot(221)
                    ax1.imshow(frame)
                    ax2 = fig.add_subplot(222)
                    ax2.hist(hist.ravel(), bins = len(hist), range=(0.0, 1.0), fc='k', ec='k')
                    fig.show()
                    a = input()
                    stop_flag = True
                else:
                    continue
        else:
            continue
                        
        cv2.imshow("Cam", frame)
        if ord('q') == cv2.waitKey(30) or stop_flag:
            break
        
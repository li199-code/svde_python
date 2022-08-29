import numpy as np
import cv2
import math

def cm_noise(img):
    single = np.mean(img, axis=2)
    avg = np.dstack((single, single, single))
    cm = img/avg
    grad = np.gradient(cm)
    # noise = np.max((grad[0],grad[1]))
    noise = grad[0]*(grad[0]>grad[1])+grad[1]*(grad[1]>grad[0])
    # cv2.imshow('0',grad[0])
    # cv2.imshow('1',grad[1])
    cv2.imshow('noise',noise)
    cv2.waitKey(-1)
    return True

if __name__ == '__main__':
    img = cv2.imread('our485/low/24.png')
    cm_noise(img)

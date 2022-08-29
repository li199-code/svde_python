from re import A
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import dtype, linalg as la
import os
from matplotlib import font_manager
# 实例化 font_manager
my_font = font_manager.FontProperties(family='SimHei', size=12)

def svde (img,p):
    HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h = HSV[:,:,0]
    s = HSV[:,:,1]
    v = HSV[:,:,2]

    # v = v+1
    v[v==0] = 1
    v1 = np.log(v)
    v2 = np.nan_to_num(v1)
    U, S, V = la.svd(v2.astype((np.float32)))
    
    # p = 1 if avg<0.1 else 0.1
    # print(p)
    gam = 1/(la.norm(S)**p)
    # plot_alpha(la.norm(S))
    # print(gam)
    S = S * gam

    Ak = np.zeros([v.shape[0], v.shape[1]])


    for i in range(S.size):
        uk = U[:, i].reshape(len(U), 1)
        vk = V[i].reshape(1, len(V))
        Ak += S[i] * np.dot(uk, vk)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    temp = normalize8(np.exp(Ak)).astype(np.uint8)
    # HSV[:,:,2] = temp
    # st1 = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

    ehcv = clahe.apply(temp)
    HSV[:,:,2] = ehcv
    st2 = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)


    return st2

def svde_v1(img):

    ## add joint upsampling
    # img_s = cv2.resize(img,dsize=None,fx=0.5,fy=0.5)
    HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    v0 = HSV[:,:,2].copy()

    scale = 0.5
    v = cv2.resize(v0,dsize=None,fx=scale,fy=scale,interpolation = cv2.INTER_NEAREST)

    # v = v+1
    v[v==0] = 1
    v1 = np.log(v)
    v2 = np.nan_to_num(v1)
    U, S, V = la.svd(v2.astype((np.float32)))
    
    gam = 1/(la.norm(S))
    S = S * gam

    Ak = np.zeros([v.shape[0], v.shape[1]])

    for i in range(S.size):
        uk = U[:, i].reshape(len(U), 1)
        vk = V[i].reshape(1, len(V))
        Ak += S[i] * np.dot(uk, vk)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    temp = normalize8(np.exp(Ak)).astype(np.uint8)
    clahetemp = clahe.apply(temp)
    temp = cv2.resize(clahetemp,dsize=(img.shape[1],img.shape[0]),interpolation = cv2.INTER_NEAREST)

    guide = cv2.ximgproc.guidedFilter(v0, temp, 10, 2, -1)
    # guide = cv2.ximgproc.jointBilateralFilter(v0, temp, 33, 5, 0)

    

    HSV[:,:,2] = guide
    st = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

    return st


def normalize8(x):
    return 255*(x - np.min(x))/(np.max(x) - np.min(x))

def plot_alpha(norm):
    x = np.arange(0, 2.01, 0.01)
    y1 = 1/norm**x
    y2 = 1-y1
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel(chr(945))
    plt.ylabel(chr(950))
    plt.title('不同'+chr(945)+'值对应的'+chr(950)+'值', fontproperties=my_font)
    plt.plot(x,y1,x,y2)
    plt.show()
    return True

# 计算彩色图的直方图
def calchist_for_rgb(img):
    # img = cv2.imread(imgname)
    histb = cv2.calcHist([img], [0], None, [256], [0, 255])
    histg = cv2.calcHist([img], [1], None, [256], [0, 255])
    histr = cv2.calcHist([img], [2], None, [256], [0, 255])
    
    plt.plot(histb, color="b")
    plt.plot(histg, color="g")
    plt.plot(histr, color="r")
    plt.savefig("result_rgba.jpg")


if __name__ == '__main__':
    # img = cv2.imread('out_stage2/kluki.png')
    # # for p in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    # #     [st1, st2] = svde(img,p)
    # #     cv2.imwrite('./paras/'+str(p)+'.png', st1)

    # calchist_for_rgb(img)

    # cv2.imshow('1',st1)
    # cv2.imshow('2',st2)
    # cv2.waitKey(-1)
    # cv2.imwrite


    # path = 'D:\BaiduNetdiskDownload\\CRENET_3C\\data\\low'
    path = 'eval15/low'
    list = os.listdir(path)
    # list = ['15.png']
    for im_na in list:
        print(im_na)
        img = cv2.imread(path + '\\' + im_na)
        v1= svde_v1(img)
        # v0 = svde(img,1)
        # out_path = path + '\\' + im_na
        cv2.imwrite('./svde_v1/'+im_na, v1)
        # cv2.imwrite('./svde/'+im_na, v0)
        # cv2.imwrite('./out_stage2/'+im_na, clahetemp)




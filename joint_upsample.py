from main import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from psnr_ssim import *
import time
from niqe.niqe import niqe

index = '02'
# ref = cv2.imread('./eval15/high/'+str(index)+'.png')
# ref = cv2.resize(ref, dsize=None, fx=2,fy=2)
img_bl = cv2.imread('./dicm/'+index+'.jpg')
# img_bl = cv2.resize(img_bl, dsize=None, fx=2,fy=2)

start = time.time()
img_oriinput = svde(img_bl, 1)
orisize_t = time.time()-start

start = time.time()
# scale = 0.5
# img_sl = cv2.pyrDown(img_bl)
# [_,img_sh] = svde(img_sl,1)
# img_bh = cv2.pyrUp(img_sh)
# imgGuidedFilter = cv2.ximgproc.guidedFilter(img_bl, img_bh, 20, 2, -1)
imgGuidedFilter = svde_v1(img_bl)
guided_t = time.time()-start

# print('ori input: {}, sampled: {}'.format(orisize_t, guided_t))
# print('ori input: {}, sampled: {}'.format(psnr(img_oriinput, ref), psnr(imgGuidedFilter, ref)))
print('ori input: {}, sampled: {}'.format(niqe(img_oriinput[:,:,0]), niqe(imgGuidedFilter[:,:,0])))

plt.subplot(221), plt.axis('off'), plt.title("Original")
plt.imshow(cv2.cvtColor(img_bl, cv2.COLOR_BGR2RGB))
plt.subplot(222), plt.axis('off'), plt.title("orisize_svde")
plt.imshow(cv2.cvtColor(img_oriinput, cv2.COLOR_BGR2RGB))
# plt.subplot(223), plt.axis('off'), plt.title("big_high")
# plt.imshow(cv2.cvtColor(img_bh, cv2.COLOR_BGR2RGB))
plt.subplot(224), plt.axis('off'), plt.title("Guided")
plt.imshow(cv2.cvtColor(imgGuidedFilter, cv2.COLOR_BGR2RGB))
plt.show()
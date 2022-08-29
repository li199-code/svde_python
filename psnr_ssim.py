import skimage
import cv2
import os
# from niqe import niqe

# pre = cv2.imread(r'D:\BaiduNetdiskDownload\CRENET_3C\data\pre\13.png')
# im = cv2.imread(r'D:\BaiduNetdiskDownload\CRENET_3C\test_results\13result.png')
# gt = cv2.imread(r'D:\BaiduNetdiskDownload\LOLdataset\eval15\high\23.png')

def psnr(img1, img2):
    return skimage.metrics.peak_signal_noise_ratio(img1, img2)

def ssim(img1, img2):
    a1 = skimage.metrics.structural_similarity(img1[:,:,0],img2[:,:,0])
    a2 = skimage.metrics.structural_similarity(img1[:,:,1],img2[:,:,1])
    a3 = skimage.metrics.structural_similarity(img1[:,:,2],img2[:,:,2])
    return (a1+a2+a3)/3

if __name__ == '__main__':
    # path1 = 'D:\\BaiduNetdiskDownload\\CRENET_3C\\test_results'
    path1 = 'D:\\svde\\svde_v1'
    path2 = 'D:\\BaiduNetdiskDownload\\LOLdataset\\eval15\\high'
    list1 = os.listdir(path1)
    # list = ['15.png']
    psn = ssi = nq = 0
    count = 0
    for im_na in list1:
        print(im_na)
        count += 1
        img = cv2.imread(path1 + '\\' + im_na)
        ref = cv2.imread(path2 + '\\' + im_na)

        # nq += niqe(img)
        psn += psnr(img,ref)
        ssi += ssim(img,ref)

    # print('niqe=' + str(nq/count))    
    print('psnr=' + str(psn/count))
    print('ssim=' + str(ssi/count))



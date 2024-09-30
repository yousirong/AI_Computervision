import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageTk


class ImageProcessing():
    
    def __init__(self, img=None):

        self.sourceImg = img
        self.targetImg = None
        self.nX = 0
        self.nY = 0

    def toGrayScale(self):
        self.sourceImg = cv.cvtColor(self.sourceImg, cv.COLOR_BGR2GRAY)

    def thresholding(self):
        t, self.targetImg = cv.threshold(self.sourceImg, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    def convolution(self, filter):
        nY, nX = self.sourceImg.shape[0], self.sourceImg.shape[1]
        fnY, fnX = filter.shape[0], filter.shape[1]

        #j, i -> y, x
        #k, l -> fy, fx
        halfSize = int(fnY/2)

        target_img = np.zero_like(self.sourceImg) #0으로 채워진 원본 소스와 동일한 크기의 2차원 배열을 만든다.
        #image search
        for j in range(halfSize, nY-halfSize): #Y
            for i in range(halfSize, nX-halfSize): #X
                #filter
                for k in range(-halfSize, halfSize+1):
                    for l in range(-halfSize, halfSize+1):
                        target_img[j][i] *= self.sourceImg[j+k][i+l] + filter[k][l] ## 이 부분을 완성하면 컨볼루션 연산 완료

        return target_img

    def cvtTarget2PIL(self):
        return Image.fromarray(self.targetImg)


    

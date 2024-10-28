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

    def toGrayScale(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return gray

    def thresholding(self, img):
        t, thresholded = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        return thresholded


    def circleDetection(self, img):
        edgeMap = self.edgeDetection(img)
        edge = (edgeMap>150)*255 #thresholding   #self.thresholding(edgeMap)

        centers, accumulator = self.houghCircleTransform(edge, 20, 23)
        print(np.min(accumulator[:,:,0]), np.max(accumulator[:,:,0]))
        self.targetImg = self.clipping(accumulator[:,:,0])
        #self.targetImg = accumulator[:,:,0]#edge

    def houghCircleTransform(self, edgeMap, min_rad, max_rad):
        # edgeMap의 높이와 너비를 구함
        nY, nX = edgeMap.shape
        # 누적 배열 초기화: 높이, 너비, 반지름의 범위에 대한 3차원 배열 생성
        accumulator = np.zeros((nY, nX, max_rad - min_rad + 1), dtype=np.uint8)

        # 에지 픽셀 순회
        for y in range(nY):
            print(print(y))  # 진행 상황을 위한 y 값 출력
            for x in range(nX):
                # 에지 픽셀인지 확인 (0보다 큰 값이면 에지로 판단)
                if edgeMap[y, x] > 0:
                    # 최소 반지름에서 최대 반지름까지의 모든 반지름에 대해 순회
                    for r in range(min_rad, max_rad + 1):
                        # 각도를 0부터 359도까지 순회하며 원의 점 계산
                        for theta in range(360):
                            # 각도를 라디안으로 변환
                            radian = theta * np.pi / 180.0
                            # 원의 x, y 좌표 계산
                            x_ = int(x - r * np.cos(radian))
                            y_ = int(y - r * np.sin(radian))

                            # 계산된 좌표가 이미지 크기 내에 있는지 확인
                            if (0 <= x_ < nX) and (0 <= y_ < nY):
                                # 누적 배열에서 해당 위치에 투표 추가
                                accumulator[y_, x_, r - min_rad] += 1

        # 누적 배열에서 최대 투표 값의 80% 이상인 좌표를 추출하여 원으로 판단
        centers = np.argwhere(accumulator > np.max(accumulator) * 0.8)

        # 검출된 원을 저장할 리스트 초기화
        circles = []
        cnt = 0
        for c in centers:
            # 중심 y, x와 반지름 r 정보 추출
            y, x, r = c
            print(cnt, ", ", y, x, r)  # 진행 상황 출력
            # 원의 중심과 반지름을 circles 리스트에 저장 (반지름은 min_rad를 더해 보정)
            circles.append((x, y, r + min_rad))
            cnt += 1

        # 검출된 원의 중심과 반지름 정보, 그리고 누적 배열 반환
        return centers, accumulator

    def edgeDetection(self, img):

        gauss = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]], dtype='float32')
        gauss = gauss/np.sum(gauss)

        dx = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype='float32')

        dy = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype='float32')

        smoothed = self.convolution(img, gauss)
        img_dx = self.convolution(smoothed, dx)
        img_dy = self.convolution(smoothed, dy)
        magnitude = np.sqrt(img_dx*img_dx + img_dy*img_dy)

        magnitude = self.clipping(magnitude) #미분결과가 음수~양수의 범위를 가지므로 절대값 + [0, 255]범위로 클립핑

        self.targetImg = magnitude

        return self.targetImg


    def convolution(self, img, filter):

        #demension
        nY, nX = img.shape[0], img.shape[1]
        fnY, fnX = filter.shape[0], filter.shape[1]

        #j, i -> y, x
        #k, l -> fy, fx
        halfSize = fnY//2

        target_img = np.zeros_like(img, dtype='float32')
        #0으로 채워진 원본 소스와 동일한 크기의 2차원 배열을 만든다. #, dtype='int16'
        #image
        for j in range(halfSize, nY-halfSize): #Y
            for i in range(halfSize, nX-halfSize): #X
                conv_value = 0
                #filter
                for k in range(-halfSize, halfSize+1):
                    for l in range(-halfSize, halfSize+1):
                        conv_value += (img[j+k][i+l] * filter[k+halfSize][l+halfSize])
                        ## 이 부분을 완성하면 컨볼루션 연산 완료

                target_img[j][i] = conv_value

        return target_img


    def clipping(self, img):
        img = np.abs(img)
        clipped = np.clip(img, 0, 255).astype('uint8')
        return clipped


    def cvtTarget2PIL(self):
        return Image.fromarray(self.targetImg)




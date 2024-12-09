import tkinter
from tkinter import filedialog
import sys
import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageTk
from tkinter import messagebox
import image_processing as ip
from ImageConverter import ImageConverter
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tkinter import messagebox


#Graphic User Interface
class MyTkinter():

    def __init__(self):
        """ 초기화 메서드로 GUI의 기본 설정과 변수를 초기화한다. """
        self.window = tkinter.Tk() # Tkinter 윈도우 생성

        self.img_source = 0 #실제, 소스, 타겟 이미지
        self.img_target = 0

        self.img_sourceTK = 0 #렌더링
        self.img_targetTK = 0

        self.ip = 0 #이미지 처리 객체
        self.cam_update_id = 0 #캠을 관리하는 업데이트 id

        self.selected_option = tkinter.IntVar(value=0) # 라디오 버튼의 선택된 옵션

        self.classNum=0 # 클래스 수

        self.imgGroupList = [] # 이미지 그룹 리스트



    def initialize(self):
        """ GUI의 초기 설정을 담당하는 메서드 """
        self.window.title("AI based Computer Vision Class")

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        window_width = 500
        window_height = 500
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height) // 2

        self.window.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}") #"500x500+200+200" #500x500 프레임웍의 크기 @ 200, 200 위치
        self.window.resizable(True, True) # 창 크기 조절 가능

         # 프레임 생성
        self.frameLeft = self.createFrame(self.window, "left")
        self.frameCenter = self.createFrame(self.window, "left")
        self.frameRight = self.createFrame(self.window, "left")
        self.frameRight2 = self.createFrame(self.window, "left")
         # 버튼 생성
        self.b1 = self.createButtonGrid(self.frameLeft, "image open", 0, 0)
        self.b2 = self.createButtonGrid(self.frameLeft, "apply", 0, 1)

        self.b3 = self.createButtonGrid(self.frameLeft, "thresholding", 1, 0)
        self.b6 = self.createButtonGrid(self.frameLeft, "circle detection", 1, 1)

        self.b4 = self.createButtonGrid(self.frameLeft, "web cam mode", 2, 0)
        self.b5 = self.createButtonGrid(self.frameLeft, "close cam", 2, 1)

        self.b7 = self.createButtonGrid(self.frameLeft, "capture", 3, 0)
        self.b8 = self.createButtonGrid(self.frameLeft, "new class", 3, 1)

        self.l1 = self.createLabel(self.frameCenter)
        self.l2 = self.createLabel(self.frameRight)
         # 버튼에 명령 연결
        self.b1.config(command=self.imgRead)
        self.b2.config(command=self.apply)
        self.b3.config(command=self.thresholding)
        self.b4.config(command=self.getCam)
        self.b5.config(command=self.stopCam)
        self.b6.config(command=self.circleDetection)

        self.b7.config(command=self.capture)
        self.b8.config(command=self.newClass)

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        #
        window_width = 600
        window_height = 600
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height) // 2

        self.window.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        self.window.resizable(True, True)

        self.frameLeft = self.createFrame(self.window, "left")
        self.frameCenter = self.createFrame(self.window, "left")
        self.frameRight = self.createFrame(self.window, "left")

        # Buttons
        self.b1 = self.createButtonGrid(self.frameLeft, "Image Open", 0, 0)
        self.b2 = self.createButtonGrid(self.frameLeft, "Train CNN", 1, 0)
        self.b3 = self.createButtonGrid(self.frameLeft, "Test CNN", 2, 0)


        self.b1.config(command=self.imgRead)
        self.b2.config(command=self.trainCNN)  # Link to trainCNN method
        self.b3.config(command=self.testCNN)  # Link to testCNN method



    def trainCNN(self):
        """ CNN 모델을 학습시키는 메서드 """
        try:
            # 경로 및 파라미터 설정
            data_dir = "dataset"
            img_size = 128
            batch_size = 32
            epochs = 10

            # 데이터 전처리 생성자
            train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
            val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

            # 학습 데이터 로드
            train_generator = train_datagen.flow_from_directory(
                data_dir,
                target_size=(img_size, img_size),
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )

            # 검증 데이터 로드
            val_generator = val_datagen.flow_from_directory(
                data_dir,
                target_size=(img_size, img_size),
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )

            # VGG16 모델 불러오기
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
            for layer in base_model.layers:
                layer.trainable = False

            x = Flatten()(base_model.output)
            # 활성화함수 relu 적용
            x = Dense(128, activation='relu')(x)
            # 절반 drop out
            x = Dropout(0.5)(x)
            predictions = Dense(train_generator.num_classes, activation='softmax')(x)

            # 모델 컴파일
            model = Model(inputs=base_model.input, outputs=predictions)
            # loss 함수 categorical_crossentropy 적용 및 Adam 옵티마이저 적용
            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

            # 모델 학습
            history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)
            model.save("cnn_model.h5")  # 모델 저장
            messagebox.showinfo("알림", "모델 학습 완료. 'cnn_model.h5'로 저장됨.")

            self.plotTrainingResults(history)  # 학습 결과 시각화
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def testCNN(self):
        """ 저장된 CNN 모델로 테스트하는 메서드 """
        try:
            model = load_model("cnn_model.h5")  # 모델 로드
            file_path = filedialog.askopenfilename(title="테스트 이미지 선택")
            img = Image.open(file_path).resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100
            class_labels = os.listdir("dataset")
            result = f"예측 클래스: {class_labels[predicted_class]}\n신뢰도: {confidence:.2f}%"
            messagebox.showinfo("예측 결과", result)
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def plotTrainingResults(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.tight_layout()
        plt.show()

    def selectionRadio(self):
        """ 라디오 버튼의 선택된 값을 반환 """
        return self.selected_option.get()

    def createRadio(self, window, name, val, r, c):
        """
        라디오 버튼을 생성하고 특정 위치에 배치하는 메서드
        """
        radio = tkinter.Radiobutton(window, text=name, variable=self.selected_option, value=val, command=self.selectionRadio)
        radio.grid(row=r, column=c)
        return radio

    def createFrame(self, window, pos, rel="solid", border=2):
        """
        새 프레임을 생성하여 특정 위치에 배치하는 메서드
        """
        frame = tkinter.Frame(window, relief=rel, bd=border)
        frame.pack(side=pos, fill="both", expand=True)
        return frame

    def createLabel(self, window):
        """
        레이블을 생성하여 특정 위치에 배치하는 메서드
        """
        label = tkinter.Label(window)
        label.pack(expand=True)
        return label

    def createLabelGrid(self, window, r, c):
        """
        그리드에 레이블을 생성하는 메서드
        """
        label = tkinter.Label(window)
        label.grid(row=r, column=c, padx=2, pady=2)
        return label

    def createButtonPlace(self, window, name, px=0, py=0, w=30, h=30):
        """
        특정 위치에 버튼을 배치하는 메서드
        """
        button = tkinter.Button(window, text=name)
        button.place(x=px, y=py, width=w, height=h)
        return button

    def createButtonGrid(self, window, name, r, c):
        """
        그리드에 버튼을 생성하는 메서드
        """
        button = tkinter.Button(window, text=name)
        button.grid(row=r, column=c)
        return button

    def buttonEvent(self, button, event):
        """
        버튼 이벤트에 명령을 연결하는 메서드
        """
        button.config(command=event)

    def newClass(self):
        """
        새로운 클래스를 추가하고 라디오 버튼을 동적으로 생성하는 메서드
        """
        self.createRadio(self.frameLeft, "class " + str(self.classNum), self.classNum, 4 + self.classNum, 0)
        self.imgGroupList.append([])  # 새로운 이미지 그룹 리스트 추가
        self.classNum += 1  # 클래스 수 증가

    def showThumbImage(self, imgArr):
        """
        이미지 목록의 썸네일을 생성하여 GUI에 표시하는 메서드
        """
        columns = 5  # 한 줄에 5개의 이미지 썸네일
        self.thumb_images = []  # 썸네일 이미지 목록

        for i in range(len(imgArr)):
            row = i // columns
            col = i % columns
            label = self.createLabelGrid(self.frameRight2, row, col)
            resized_img = cv.resize(imgArr[i], (50, 50))  # 이미지 크기를 50x50으로 조절
            photo = ImageConverter.toPhotoImage(resized_img)  # 이미지 변환
            self.thumb_images.append(photo)  # 참조 유지
            label.config(image=photo)  # 라벨에 이미지 추가

    def capture(self):
        """
        웹캠의 이미지를 캡처하여 이미지 그룹에 추가하는 메서드
        """
        self.imgGroupList[self.selectionRadio()].append(self.img_target)  # 선택된 클래스에 이미지 추가
        for i in range(len(self.imgGroupList)):
            print(i, len(self.imgGroupList[i]))  # 각 클래스에 추가된 이미지 수 출력
        self.showThumbImage(self.imgGroupList[self.selectionRadio()])  # 썸네일 표시

    def getCam(self):
        """
        웹캠을 실행하여 GUI에 영상을 출력하는 메서드
        """
        cap = cv.VideoCapture(0)  # 웹캠 연결

        if not cap.isOpened():
            messagebox.showinfo("Info", "카메라를 열 수 없습니다.")
            return

        def update_frame():
            """
            10ms마다 프레임을 업데이트하는 내부 함수
            """
            ret, img = cap.read()
            if ret:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # BGR을 RGB로 변환
                self.img_target = img
                self.img_source = ImageConverter.toPIL(self.img_target)
                self.img_sourceTK = ImageConverter.toPhotoImage(self.img_source)
                self.l1.config(image=self.img_sourceTK)  # GUI에 이미지 표시
            self.cam_update_id = self.window.after(10, update_frame)  # 10ms 후 재호출

        update_frame()  # 업데이트 시작

    def stopCam(self):
        """
        웹캠을 중지하는 메서드
        """
        if self.cam_update_id is not None:
            self.window.after_cancel(self.cam_update_id)  # 업데이트 취소
            self.cam_update_id = None

        if self.cap is not None:
            self.cap.release()  # 웹캠 연결 해제
            self.cap = None

    def imgRead(self):
        """
        파일 대화 상자를 통해 이미지를 불러오는 메서드
        """
        file_path = filedialog.askopenfilename(title="파일 선택")
        img = cv.imread(file_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.img_source = img
        self.img_sourceTK = ImageConverter.toPhotoImage(img)  # 이미지를 tkinter 포맷으로 변환
        self.l1.config(image=self.img_sourceTK)  # 이미지 표시

    def thresholding(self):
        """
        이미지를 이진화(Thresholding)하는 메서드
        """
        img = self.ip.sourceImg
        gray_img = self.ip.toGrayScale(img)  # 그레이스케일 변환
        thresholded = self.ip.thresholding(gray_img)  # 이진화 처리
        self.ip.targetImg = thresholded
        self.img_target = self.ip.targetImg
        messagebox.showinfo("Info", "처리 완료")

    def circleDetection(self):
        """
        원을 검출하는 메서드
        """
        img = self.ip.sourceImg
        gray_img = self.ip.toGrayScale(img)  # 그레이스케일 변환
        self.ip.circleDetection(gray_img)  # 원 검출
        self.img_target = self.ip.targetImg
        messagebox.showinfo("Info", "원 검출 완료")

    def apply(self):
        """
        전처리된 이미지를 GUI에 표시하는 메서드
        """
        self.img_target = ImageConverter.toPIL(self.img_target)  # PIL 포맷으로 변환
        self.img_targetTK = ImageConverter.toPhotoImage(self.img_target)  # tkinter 포맷으로 변환
        self.l2.config(image=self.img_targetTK)  # 이미지 표시

    def render(self):
        """
        메인 루프를 실행하는 메서드
        """
        self.window.mainloop()


# 프로그램의 메인 실행 코드
if __name__ == "__main__":
    tk = MyTkinter()  # MyTkinter 객체 생성
    tk.initialize()  # GUI 초기화
    tk.render()  # 메인 루프 실행


'''
3/3 ━━━━━━━━━━━━━━━━━━━━ 12s 2s/step - accuracy: 0.5666 - loss: 1.0213 - val_accuracy: 0.4706 - val_loss: 0.7312
Epoch 2/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - accuracy: 0.8293 - loss: 0.5150 - val_accuracy: 1.0000 - val_loss: 0.0196
Epoch 3/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - accuracy: 0.9887 - loss: 0.0435 - val_accuracy: 1.0000 - val_loss: 0.0287
Epoch 4/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 80ms/step - accuracy: 0.9472 - loss: 0.0851 - val_accuracy: 1.0000 - val_loss: 0.0136
Epoch 5/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 87ms/step - accuracy: 0.9887 - loss: 0.0286 - val_accuracy: 1.0000 - val_loss: 0.0027
Epoch 6/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - accuracy: 1.0000 - loss: 0.0124 - val_accuracy: 1.0000 - val_loss: 8.9454e-04
Epoch 7/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 84ms/step - accuracy: 1.0000 - loss: 0.0021 - val_accuracy: 1.0000 - val_loss: 4.9416e-04
Epoch 8/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 73ms/step - accuracy: 1.0000 - loss: 0.0085 - val_accuracy: 1.0000 - val_loss: 2.9195e-04
Epoch 9/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 119ms/step - accuracy: 1.0000 - loss: 0.0018 - val_accuracy: 1.0000 - val_loss: 2.2559e-04
Epoch 10/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 114ms/step - accuracy: 1.0000 - loss: 0.0130 - val_accuracy: 1.0000 - val_loss: 2.1099e-04
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
'''
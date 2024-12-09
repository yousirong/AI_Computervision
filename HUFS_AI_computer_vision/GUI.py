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


# =====================
# Graphic User Interface
# =====================
class MyTkinter():

    def __init__(self):
        """ 초기화 메서드 - instance 클래스 변수 초기화 """
        self.window = tkinter.Tk()  # tkinter 윈도우 창 생성

        self.img_source = 0  # 원본 이미지 변수 초기화
        self.img_target = 0  # 전처리 후 이미지 변수 초기화

        self.img_sourceTK = 0  # tkinter용 원본 이미지 변수 초기화
        self.img_targetTK = 0  # tkinter용 전처리 이미지 변수 초기화

        self.ip = 0  # 이미지 처리 객체 (ImageProcessing 인스턴스) 변수 초기화
        self.cam_update_id = 0  # 카메라 업데이트 ID 변수 초기화

        self.selected_option = tkinter.IntVar(value=0)  # 라디오 버튼 값 변수 초기화
        self.classNum = 0  # 클래스 개수 변수 초기화
        self.imgGroupList = []  # 이미지 그룹 리스트 변수 초기화

    def initialize(self):
        """ GUI 초기화 및 위젯 배치(버튼, 레이블, 프레임) 생성 """
        self.window.title("AI based Computer Vision Class")  # 창 제목 설정

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        window_width = 500
        window_height = 500
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height) // 2

        self.window.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        self.window.resizable(True, True)  # 창 크기 조절 가능

        # 총 4개의 프레임 생성
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

        # 레이블 생성
        self.l1 = self.createLabel(self.frameCenter)
        self.l2 = self.createLabel(self.frameRight)

        # 버튼의 동작 연결
        self.b1.config(command=self.imgRead)
        self.b2.config(command=self.apply)
        self.b3.config(command=self.thresholding)
        self.b4.config(command=self.getCam)
        self.b5.config(command=self.stopCam)
        self.b6.config(command=self.circleDetection)
        self.b7.config(command=self.capture)
        self.b8.config(command=self.newClass)

    def trainCNN(self):
        """ CNN 학습 함수 컴퓨팅 파워 낮은 gpu used : titan xp vram 12G"""
        try:
            data_dir = "dataset"  # 데이터셋 디렉터리
            img_size = 128  # 이미지 크기
            batch_size = 32  # 배치 크기
            epochs = 10  # 에포크 수

            train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
            val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

            # 데이터 로드
            train_generator = train_datagen.flow_from_directory(
                data_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical', subset='training'
            )
            val_generator = val_datagen.flow_from_directory(
                data_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical', subset='validation'
            )
            # VGG 16 사용
            # includ_top = False -> Fully Connected Layer(FC Layer)를 제거하고 특성 추출기로 사용
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
            for layer in base_model.layers:
                layer.trainable = False  # VGG16의 레이어는 동결

            # 커스텀 레이어 추가
            x = Flatten()(base_model.output)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(train_generator.num_classes, activation='softmax')(x)
            # 옵티마이저는 Adam 사용 0.001
            model = Model(inputs=base_model.input, outputs=predictions)
            # 손실 함수로 categorical_crossentropy 사용
            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)
            model.save("cnn_model.h5")  # 모델 저장

            self.plotTrainingResults(history)  # 학습 결과 시각화
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def testCNN(self):
        """ CNN 테스트 함수  """
        try:
            # 파일 대화 상자를 띄워 테스트에 사용할 이미지를 선택
            model = load_model("/home/juneyonglee/Desktop/vision_class/cnn_model.h5")
            file_path = filedialog.askopenfilename(title="Select a Test Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")])
            if not file_path:
                return

            img_size = 128
            img = Image.open(file_path).resize((img_size, img_size))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            # 모델의 예측 결과에서 가장 높은 확률의 클래스를 찾기
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100
            class_labels = os.listdir("dataset")

            result = f"Predicted Class: {class_labels[predicted_class]}\nConfidence: {confidence:.2f}%"
            messagebox.showinfo("Prediction Result", result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plotTrainingResults(self, history):
        """ 학습 결과 시각화 메서드 """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(len(acc))
        # 모델의 학습 이력(acc, loss)을 시각화
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

    def imgRead(self):
        """ 이미지 불러오기오고 읽는 메서드 """
        file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("img files", "*.jpg"), ("All files", "*.*")])
        img = cv.imread(file_path)
        # 파일 창을 열어서 탐색
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.img_source = img

    def apply(self):
        """ 전처리된 이미지를 GUI에 표시 및 출력하는 메서드 """
        # 전처리된 이미지를 PIL 및 tkinter 포맷으로 변환 후 레이블에 표시
        self.img_target = ImageConverter.toPIL(self.img_target)
        self.img_targetTK = ImageConverter.toPhotoImage(self.img_target)
        self.l2.config(image=self.img_targetTK)

    def getCam(self):
        """ 웹캠 영상을 Tkinter GUI에 출력하는 메서드 """
        cap = cv.VideoCapture(0)

    def stopCam(self):
        """ 웹캠 중지 메서드 """
        # 카메라 스트리밍을 중지
        if self.cam_update_id is not None:
            self.window.after_cancel(self.cam_update_id)

    def render(self):
        """ GUI 메인 루프 실행 메서드 """
        self.window.mainloop()

# Main 실행부
if __name__ == "__main__":
    tk = MyTkinter()
    tk.initialize()
    tk.render()

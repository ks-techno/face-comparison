# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, BaseModel, File


print('file reading on first line')

import numpy as np
from keras_facenet import FaceNet

from mtcnn import MTCNN

import cv2 as cv
from flask import jsonify
from PIL import Image
# from urllib.request import urlopen








class Output(BaseModel):
    success: bool
    is_same_person: bool
    confidence_score: float


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.detector = MTCNN()
        self.embedder = FaceNet()
    
    def expand_box(self, box, scaling_factor, image_width, image_height):
        x, y, width, height = box

        # Calculate new dimensions
        new_width = width * scaling_factor
        new_height = height * scaling_factor

        # Calculate new (x, y) coordinates
        new_x = x - (new_width - width) / 2
        new_y = y - (new_height - height) / 2

        # Ensure new coordinates and dimensions are within bounds
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_width = min(image_width - new_x, new_width)
        new_height = min(image_height - new_y, new_height)

        expanded_box = (int(new_x+1), int(new_y+1), int(new_width-1), int(new_height-1))
        return expanded_box

    def faces_manual(self, img):
        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces)>0:
            box = self.expand_box( faces[0], 1.2, img.shape[1], img.shape[0] )
        else:
            box = (0, 0, img.shape[1], img.shape[0])
        return box

    def predict(
        self,
        url1: File = Input(description="input image1"),
        url2: File = Input(description="input image2")
    ) -> Output:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        images = []
        # try:
        #     images.append(np.array(Image.open(url1)))
        #     images.append(np.array(Image.open(url2)))
        #     print('url needs not to be loaded')
        # except Exception as exp:
        #     print('url needs to be loaded')

        images.append(np.array(Image.open(url1))[:,:,:3])
        images.append(np.array(Image.open(url2))[:,:,:3])
        faces = [self.detector.detect_faces(img) for img in images]
        faces = [self.expand_box(i[0]['box'], 1.2, j.shape[1], j.shape[0]) if len(i) > 0 else self.faces_manual(j) for i, j in
                    zip(faces, images)]
        images = [img[y:y + height, x:x + width] for (x, y, width, height), img in zip(faces, images)]

        embeddings = self.embedder.embeddings(images)
        confidence = 1.0 - (self.embedder.compute_distance(embeddings[0], embeddings[1]) - 0.2)
        confidence = max(confidence, 0.0)
        confidence = min(confidence, 1.0)
        print(confidence)
        # return confidence
        if confidence >= 0.75:
            return Output(success= True,
                            is_same_person= True,
                            confidence_score= confidence)
        else:
            return Output(success= True,
                            is_same_person= False,
                            confidence_score= confidence)

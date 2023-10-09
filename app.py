print('file reading on first line')
from flask import Flask, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from keras_facenet import FaceNet
embedder = FaceNet()
from mtcnn import MTCNN
detector = MTCNN()
import cv2 as cv
from flask import jsonify
from PIL import Image
from urllib.request import urlopen

app = Flask(__name__)
auth = HTTPBasicAuth()

users = {
    "face": generate_password_hash("face2face")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and \
            check_password_hash(users.get(username), password):
        return username

def expand_box(box, scaling_factor, image_width, image_height):
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

def faces_manual(img):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces)>0:
        box = expand_box( faces[0], 1.2, img.shape[1], img.shape[0] )
    else:
        box = (0, 0, img.shape[1], img.shape[0])
    return box


@app.route('/compare-faces-file', methods=['POST'])
@auth.login_required
def file():
    try:
        if "multipart/form-data" not in request.content_type:
            error_message = "Unsupported Media Type. Please send data in multipart/form-data."
            return jsonify({
                "success": False,
                "error_message": error_message
            }), 415
        if 'image_file_1' not in request.files.keys():
            return jsonify({
                "success": False,
                "error_message": f"image_file_1 not found in request"
            }), 400
        if 'image_file_2' not in request.files.keys():
            return jsonify({
                "success": False,
                "error_message": f"image_file_2 not found in request"
            }), 400
        images = []
        file1 = request.files.get('image_file_1')
        file2 = request.files.get('image_file_2')
        if ('.jpg' in file1.filename.lower() or '.jpeg' in file1.filename.lower() or '.png' in file1.filename.lower()) and \
                ('.jpg' in file2.filename.lower() or '.jpeg' in file2.filename.lower() or '.png' in file2.filename.lower()):
            try:
                images.append(np.array(Image.open(file1)))
            except Exception as exp:
                return jsonify({
                    "success": False,
                    "error_message": f"image_file_1 could not be loaded due to ({exp}) error"
                }), 500
            try:
                images.append(np.array(Image.open(file2)))
            except Exception as exp:
                return jsonify({
                    "success": False,
                    "error_message": f"image_file_2 could not be loaded due to ({exp}) error"
                }), 500

            try:
                faces = [detector.detect_faces(img) for img in images]
                faces = [expand_box(i[0]['box'], 1.2, j.shape[1], j.shape[0]) if len(i) > 0 else faces_manual(j) for i, j in
                         zip(faces, images)]
                images = [img[y:y + height, x:x + width] for (x, y, width, height), img in zip(faces, images)]

                embeddings = embedder.embeddings(images)
                confidence = 1.0 - (embedder.compute_distance(embeddings[0], embeddings[1]) - 0.2)
                confidence = max(confidence, 0.0)
                confidence = min(confidence, 1.0)
                if confidence >= 0.75:
                    return jsonify({"success": True,
                                    "is_same_person": True,
                                    "confidence_score": confidence})
                else:
                    return jsonify({"success": True,
                                    "is_same_person": False,
                                    "confidence_score": confidence})
            except Exception as exp:
                return jsonify({
                    "success": False,
                    "error_message": f"In face comparison an unexpected error ({exp}) occured"
                }), 500
        else:
            return jsonify({
                        "success": False,
                        "error_message": f"Only jpg, jpeg and png files are acceptable"
                        }), 400
    except Exception as exp:
        return jsonify({
            "success": False,
            "error_message": f"An unknown error ({exp}) occured."
        }), 500

@app.route('/compare-faces-url', methods=['POST'])
@auth.login_required
def index():
    try:
        if request.content_type != "application/json":
            error_message = "Unsupported Media Type. Please send data in JSON format."
            return jsonify({
                "success": False,
                "error_message": error_message
            }), 415
        if 'image_url_1' not in request.get_json().keys():
            return jsonify({
                "success": False,
                "error_message": f"image_url_1 not found in request"
            }), 400
        if 'image_url_2' not in request.get_json().keys():
            return jsonify({
                "success": False,
                "error_message": f"image_url_2 not found in request"
            }), 400
        url1, url2 = request.get_json().get('image_url_1'), request.get_json().get('image_url_2')
        images = []
        try:
            response = urlopen(url1)
            image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
            images.append(cv.imdecode(image_array, cv.IMREAD_COLOR))
        except Exception as exp:
            return jsonify({
                        "success": False,
                        "error_message": f"image_url_1 could not be loaded due to ({exp}) error"
                        }), 500
        try:
            response = urlopen(url2)
            image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
            images.append(cv.imdecode(image_array, cv.IMREAD_COLOR))
        except Exception as exp:
            return jsonify({
                        "success": False,
                        "error_message": f"image_url_2 could not be loaded due to ({exp}) error"
                        }), 500
        try:
            faces = [detector.detect_faces(img) for img in images]
            faces = [expand_box( i[0]['box'], 1.2, j.shape[1], j.shape[0] ) if len(i)>0 else faces_manual(j) for i,j in zip(faces,images) ]
            images = [ img[y:y+height, x:x+width] for (x,y,width,height),img in zip(faces,images) ]

            embeddings = embedder.embeddings(images)
            confidence = 1.0 - (embedder.compute_distance(embeddings[0], embeddings[1])-0.2)
            confidence = max(confidence,0.0)
            confidence = min(confidence, 1.0)
            if confidence >= 0.75:
                return jsonify({"success": True,
                                "is_same_person": True,
                                "confidence_score": confidence })
            else:
                return jsonify({"success": True,
                                "is_same_person": False,
                                "confidence_score": confidence})
        except Exception as exp:
            return jsonify({
                        "success": False,
                        "error_message": f"In face comparison an unexpected error ({exp}) occured"
                        }), 500
    except Exception as exp:
        return jsonify({
            "success": False,
            "error_message": f"An unknown error ({exp}) occured."
        }), 500

@app.route('/', methods=['GET'])
@auth.login_required
def status():
    return 'running'

if __name__ == '__main__':
   app.run(host = '0.0.0.0', debug = False, port=5011)



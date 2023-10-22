# setting path
import sys
sys.path.append('../')
from flask import Blueprint, jsonify, request, g
from pydantic import BaseModel, validator, Extra, ValidationError
import numpy as np
# import os
# from flask import redirect, url_for
# from flask_cors import CORS, cross_origin
from runObjectness import run_objectness
import cv2
from defaultParams import default_params
# from drawBoxes import draw_boxes, save_boxes
# from computeObjectnessHeatMap import compute_objectness_heat_map
# import time
from PIL import Image
import config
import json
# import io
# from flask import Response
from computeFunction import *

from utils.utils import generic_api_requests
from flask import jsonify
from sklearn.metrics.pairwise import cosine_similarity

from spm1 import *
# from LOCG import *
from PFW import *



# Create two constant. They direct to the app root folder and logo upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_FOLDER = os.path.join(APP_ROOT, '/static/')

activities = Blueprint(name="activities", import_name=__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return [item.tolist() for item in obj]
        return json.JSONEncoder.default(self, obj)
    
def _sparse_sim_matrix(S, M, nb):
    copied_matrix = np.copy(S)
    assign_box = [int(id / M) for id in range(nb)]
    for i in range(nb):
        for j in range(nb):
            if assign_box[i] == assign_box[j]:
                copied_matrix[i, j] = 0
    return copied_matrix

def resize_ratio(img, name):
    im = Image.open(img)
    width, height = im.size
    ratio = int(min(width, height) / 100)
    im = im.resize((width//ratio, height//ratio))
    im.save(name)
    return im

def extract_boxes(M, params, data_dir, imgs):
    box_data = []
    box_coordinates = []
    raw_data = []
    for img in imgs:
        img_id = data_dir + img
        # print("img id {}".format(img_id))
        img_example = cv2.imread(img_id, cv2.IMREAD_COLOR)[:, :, ::-1]
        # print("img shape {}".format(img_example.shape))
        boxes = run_objectness(img_example, M, params)
        box_coordinates = box_coordinates + boxes.tolist()
        # print(img)
        for box in boxes:
            xmin, ymin, xmax, ymax, _ = round_box(box)
            raw_img = img_example[ymin:ymax, xmin:xmax, :]
            gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY) 
            box_data.append(gray_img)
            raw_data.append(raw_img)
    return box_data, box_coordinates, raw_data

def get_box_prior(bxs):
    saliency = saliency_map_from_set(bxs)
    # print(saliency)
    v = []
    for s in saliency:
        # In paper, it is weighted by score, will fix later
        v.append(np.mean(s))
    return np.array(v)
    
class ActivityCreationFilterInput(BaseModel, extra=Extra.forbid):
    name: str

    @validator("name")
    def validate_name(cls, value):
        if value == "":
            raise ValueError("can not be empty")
        return value
    
def split_array(arr):
    list_arr = str(arr).split(',')
    i = 0
    result = []
    t = []
    while i < len(list_arr):
        k = i % 5
        if k == 4:
            t.append(list_arr[i])
            result.append(tuple(t))
            t = []
        else:
            t.append(list_arr[i])
        i = i + 1
    return result

def boxes_data_from_img(boxes, img_id):
    box_data = []
    img_data = cv2.imread(img_id, cv2.COLOR_BGR2RGB)
    for box in boxes:
        xmin, ymin, xmax, ymax, score = round_box(box)
        box_data.append(img_data[ymin:ymax, xmin:xmax, :])
    return box_data


@activities.route("/", methods=["POST"], strict_slashes=False)
def create_activity():
    try:

        request_body = json.loads(
            ActivityCreationFilterInput(**request.get_json()).json()
        )

        is_success, response = generic_api_requests(
            "post", config.URL_ACTIVITIES, request_body
        )

        response_body = {
            "success": is_success,
            "data": response["json"] if is_success else {"message": str(response)},
        }

        return jsonify(response_body)

    except ValidationError as e:
        print(g.execution_id, " VALIDATION ERROR", e)

        response = {
            "success": 0,
            "data": {"message": ("RTFM {}".format(e))},
        }

        return response, 400

    except Exception as error:

        response_body = {
            "success": 0,
            "data": {"message": "Error : {}".format(error)},
        }

        return jsonify(response_body), 400

@activities.route("/predict", methods=["POST", "GET"], strict_slashes=False)
def lasso_predict():
    args = request.form.getlist("features")
    # features = request.args["features"]
    features = [float(i) for i in args[0].split(',')]
    X = np.array(features, dtype=float)
    W = np.fromfile('static/Lasso_W.txt')
    b = np.fromfile('static/Lasso_b.txt')
    prediction = np.dot(W.T, X) + b
    return jsonify(prediction=prediction[0], safe=False)


@activities.route("/fun/", methods=["POST", "GET"], strict_slashes=False)
def fun():
    return "123"


@activities.route("/objectness", methods=["POST", "GET"], strict_slashes=False)
def objectness():
    # return '234'
    print(request.files)
    image1 = request.files["image1"]
    image2 = request.files["image2"]
    image3 = request.files["image3"]
    image4 = request.files["image4"]

    # im = PIL.Image.open("email.jpg"))
    # width, height = im.size
    # im = im.resize((width//2, height//2))
    image1 = resize_ratio(image1, 'img1.jpg')
    image2 = resize_ratio(image2, 'img2.jpg')
    image3 = resize_ratio(image3, 'img3.jpg')
    image4 = resize_ratio(image4, 'img4.jpg')

    num_images = 4    
    num_box_per_image = 10
    imgs = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']  
    M = 10
    i = 0
    params = default_params('.')
    boxes_data, box_coordinates, raw_data = extract_boxes(M, params, '', imgs)
    variable_index = np.ones(shape=(40, 3))
    for i in range(num_images):
        for j in range(num_box_per_image):
            variable_index[i, :] = np.array([1, i, j])

    boxes_data = [np.uint8(cv2.resize(img,(28,28))*255) for img in boxes_data]

    # raw_data = [np.uint8(cv2.resize(img,(28,28))*255) for img in raw_data]
    box_prior = get_box_prior(raw_data)


    np.save('boxes_data.npy', boxes_data)

    with open('box_coordinates.npy', 'wb') as f:
        np.save(f, np.array(box_coordinates))

    with open('box_prior.npy', 'wb') as f:
        np.save(f, np.array(box_prior))

    return json.dumps({'variable_index': variable_index}, cls=NumpyEncoder)


@activities.route("/getFeatures", methods=["POST", "GET"], strict_slashes=False)
def ectract_feature():
    # boxes = split_array(request.form.get("boxes"))
    # imgs = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']  
    mu = 0.6
    lamda = 0.001
    M = 10
    params = default_params('.')
    params.cues = ['SS']
    # boxes_data, box_coordinates, raw_data = extract_boxes(M, params, '', imgs)
    with open('boxes_data.npy', 'rb') as f:
        boxes_data = np.load(f)

    # with open('box_coordinates.npy', 'rb') as f:
    #     box_coordinates = np.load(f)

    with open('box_prior.npy', 'rb') as f:
        box_prior = np.load(f)

    # x_train = [np.uint8(cv2.resize(img,(28,28))*255) for img in boxes_data]
    x_train = boxes_data
    x_train = np.asarray(x_train)

    VOC_SIZE = 1000
    PYRAMID_LEVEL = 1
    DSIFT_STEP_SIZE = 4

    x_train_feature = [extract_DenseSift_descriptors(img) for img in x_train]

    x_train_kp, x_train_des = zip(*x_train_feature)

    codebook = build_codebook(x_train_des, VOC_SIZE)

    x_train = [spatial_pyramid_matching(x_train[i],
                                        x_train_des[i],
                                        codebook,
                                        level=PYRAMID_LEVEL)
                                        for i in range(len(x_train))]
    # nb: total number of boxes
    # nb: total number of boxes
    nb = len(boxes_data)

    S = cosine_similarity(x_train)
    similarity_matrix = _sparse_sim_matrix(S, M, nb)

    # box_prior = get_box_prior(raw_data)

    if box_prior.ndim == 1:
        box_prior = np.expand_dims(box_prior, 1)

    L = normalize_laplacian(similarity_matrix)

    k = 2

    vector_one = np.ones(shape=(nb,1))

    dim = VOC_SIZE

    I = np.identity(nb)

    central_matrix = I - (1/nb) * np.dot(vector_one, vector_one.T)

    b = np.ones(shape=(nb, nb))

    Identity = np.identity(dim)

    A = discriminative_optimial(central_matrix, x_train, nb, Identity, k)

    nb = len(x_train)

    A = L + mu*A

    b = lamda * np.log(box_prior)

    with open('A.npy', 'wb') as f:
        np.save(f, A)

    with open('b.npy', 'wb') as t:
        np.save(t, b)

    return jsonify({"result":'ok'})


@activities.route("/optimize", methods=["POST", "GET"], strict_slashes=False)
def optimize():
    # A = request.form.get('A') 
    # b = request.form.get('b') 

    with open('A.npy', 'rb') as f:
        A = np.load(f)

    with open('b.npy', 'rb') as t:
        b = np.load(t)

    variable_index = []

    num_images = 4

    num_box_per_image = 10

    for i in range(1, num_images + 1):
            for j in range(1, num_box_per_image + 1):
                variable_index.append(np.array([1, i, j]))

    variable_index = np.vstack(variable_index)

    opts = PARAM()

    x_0, S_0, alpha_0, ids = init_images(variable_index)

    opts.Tmax  = 1000 # max number of iteration
    opts.TOL   = 1e-8 # tolerance for convergence
    opts.verbose = True

    opts.pureFW = 0
    x_t,f_t, resPairFW = PFW(x_0, S_0, alpha_0, A, b, solver_images, cost_fun, ids, opts)


    return jsonify(x_t= x_t.tolist(), safe=False)
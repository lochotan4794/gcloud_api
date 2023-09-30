import config
import json
import sys
import io
# setting path
sys.path.append('../')
from flask import Blueprint, jsonify, request, g
from pydantic import BaseModel, validator, Extra, ValidationError
import numpy as np
import os
from flask import redirect, url_for
from flask_cors import CORS, cross_origin
from runObjectness import run_objectness
import cv2
from defaultParams import default_params
from drawBoxes import draw_boxes, save_boxes
from computeObjectnessHeatMap import compute_objectness_heat_map
import time
from PIL import Image
from flask import Response
from computeFunction import *
from PCG import *


from utils.utils import generic_api_requests

activities = Blueprint(name="activities", import_name=__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class ActivityCreationFilterInput(BaseModel, extra=Extra.forbid):
    name: str

    @validator("name")
    def validate_name(cls, value):
        if value == "":
            raise ValueError("can not be empty")
        return value


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
    image = request.files["image"].read()      
    if image:
        image  = io.BytesIO(image)
        image = np.array(Image.open(image))
        img_example = image[:, :, ::-1]
        params = default_params('.')
        boxes = run_objectness(img_example, 10, params)
        return json.dumps({'boxes': boxes}, cls=NumpyEncoder)
    return jsonify(image_pth="", safe=False)


@activities.route("/ectract_feature", methods=["POST", "GET"], strict_slashes=False)
def ectract_feature():
    boxes = request.files["image"].read()      
    train_data = [cv2.imread(img, cv2.COLOR_BGR2RGB) for img in boxes]
    x_train = computeSIFT(train_data)
    all_train_desc = []
    for i in range(len(x_train)):
        for j in range(x_train[i].shape[0]):
            all_train_desc.append(x_train[i][j,:])
    all_train_desc = np.array(all_train_desc)
    k = 10
    Level = 2
    kmeans = clusterFeatures(all_train_desc, k)
    train_histo = getHistogramSPM(Level, train_data, kmeans, k)
    X = train_histo
    similarity_matrix = np.corrcoef(X)
    box_prior = box_prior(train_data)
    L = normalize_laplacian(similarity_matrix)
    nb = len(x_train)
    k = 2
    vector_one = np.ones(shape=(nb,1))
    dim = X.shape[1]
    I = np.identity(nb)
    central_matrix = I - (1/nb) * np.dot(vector_one, vector_one.T)
    mu = 0.5
    lamda = 0.1
    b = np.ones(shape=(nb, nb))
    Identity = np.identity(dim)
    A = discriminative_optimial(central_matrix, X, nb, Identity, k)
    A = L + mu*A
    b = lamda * np.log(box_prior)
    return jsonify({'A': A, 'b': b}, safe=False)


@activities.route("/optimize", methods=["POST", "GET"], strict_slashes=False)
def optimize():
    annots = request.files["annots"].read()      
    A = annots['A']
    b = annots['b']
    var_index = annots['var_index']
    edge_index = annots['edge_index']
  
    x_0, S_0, alpha_0, ids = init_images(var_index)

    opts.Tmax  = 2000 # max number of iteration
    opts.TOL   = 1e-8 # tolerance for convergence
    opts.verbose = True

    opts.pureFW = 0
    x_t,f_t, resPairFW = PFW(x_0, S_0, alpha_0, A, b, solver_images, cost_fun, ids, opts)

    return jsonify({'x_t': x_t}, safe=False)
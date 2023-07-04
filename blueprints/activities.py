import config
import json

from flask import Blueprint, jsonify, request, g
from pydantic import BaseModel, validator, Extra, ValidationError
import numpy as np
import os
from flask import redirect, url_for

from utils.utils import generic_api_requests

activities = Blueprint(name="activities", import_name=__name__)


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
from flask import Flask
from flask_cors import CORS
from app.model_predict.model import model as model_predict

def create_app():
    app = Flask(__name__)

    app.register_blueprint(model_predict, url_prefix='/model')

    return app
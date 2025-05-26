from flask import Blueprint

model = Blueprint('main', __name__)

from . import model
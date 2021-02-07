from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import spacy
import glob
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_sm')

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
predictor.predict(
    sentence="Did Uriah honestly think he could beat the game in under three hours?."
)
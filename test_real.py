import time
import logging
import pathlib
from copy import deepcopy
import subprocess
import json
import tqdm

from ds_classifier import DSClassifier
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.data_stream import DataStream
import numpy as np
import pandas as pd
from pyinstrument import Profiler

from skika.evaluation.inspect_recurrence import InspectPrequential

options = {
    'generator': 'reverse_RBF',
    'difficulty': 2,
    'gen_seed': 4,
    'seed': 36,
    'positions': [5000, 12000, 18000],
    # 'positions': [5000, 10000],
    'widths': [1, 4000, 1],
    # 'widths': [1, 1],
    'run_length': 19000,
    "window": 5000,
    "sensitivity": 0.05,
    "poisson": 10,
    "num_alternative_states": 5,
    "conf_sensitivity_drift": 0.05,
    "conf_sensitivity_sustain": 0.125,
    "alt_test_length": 2000,
    "alt_test_period": 2000,
}

classifier = DSClassifier(
    learner=lambda : deepcopy(HoeffdingTree()),
    allow_backtrack=False,
    window = options["window"],
    sensitivity = options["sensitivity"],
    poisson = options["poisson"],
    num_alternative_states = options["num_alternative_states"],
    conf_sensitivity_drift = options["conf_sensitivity_drift"],
    conf_sensitivity_sustain = options["conf_sensitivity_sustain"],
    alt_test_length = options["alt_test_length"],
    alt_test_period = options["alt_test_period"]
)

df = pd.read_csv(pathlib.Path.cwd() / 'RawData' / 'R' / 'stream-RangioraClean_dataset.csv')
length = df.shape[0]
mask = list(df['mask'])
df = df.drop('mask', axis = 1)
print(df.head())

stream = DataStream(df)
stream.prepare_for_use()

ex = 0
right = 0
wrong = 0
for ex in tqdm.tqdm(range(length)):
    mask_val = mask[ex]
    X,y = stream.next_sample()
    p = classifier.predict(X)
    classifier.partial_fit(X, y, masked = mask_val == 1)
    if p[0] == y[0]:
        right += 1
    else:
        wrong += 1
print(right / (right + wrong))
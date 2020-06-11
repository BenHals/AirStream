from copy import deepcopy
import time
import numpy as np
import math
import logging
import scipy.stats
from scipy.interpolate import Rbf
from skmultiflow.utils import get_dimensions, check_random_state
from systemStats import systemStats
from sklearn.gaussian_process import GaussianProcessClassifier

gaussian_cache = {}
avg_distance_cache = None
def levelize(v, bins):
    # return np.digitize(v, [50, 100, 150, 200])
    # return np.digitize(v, [12, 35.5, 55.5, 150.5, 250.5])
    b = bins if not bins is None else [12, 35.5, 55.5, 150.5, 250.5]
    leveled = np.digitize(v, b)
    return leveled

def OK_interpolation(X, spacial_pattern, bins, avg_distance_cache = None):
    window_x_GP = []
    window_y = []
    y_vals = set()
    for i,feature in enumerate(X):
        t_x, t_y, t_t, t_d = spacial_pattern[i]
        window_x_GP.append(np.array([t_x, t_y, t_t]))
        y = levelize(feature, bins)
        window_y.append(y)
        y_vals.add(y)
    if len(list(y_vals)) == 1:
        return list(y_vals)[0], GaussianProcessClassifier()
    train_X = np.vstack(window_x_GP)

    classifier = GaussianProcessClassifier()
    classifier.fit(train_X, window_y)
    return classifier.predict(np.array([spacial_pattern[-1][0], spacial_pattern[-1][1], 0]).reshape(1, -1))[0], classifier
def temporal_interpolation(X, spacial_pattern, bins, avg_distance_cache = None):
    # level = levelize(X[-1], bins)
    level = X[-1]
    logging.debug(f"level: {level}")
    return level, level
def linear_interpolation(X, spacial_pattern, bins, avg_distance_cache = None):
    if spacial_pattern is None:
        logging.debug(f"No spacial pattern so defaulting to avg")
        return linear_interpolation_no_spacial(X, bins), spacial_pattern
    X = X[:(len(X) - 1)//2]
    # spacial_pattern = spacial_pattern[:(len(X) - 1)//2]
    # print([x[-1] for x in spacial_pattern])
    total_distance = sum([1 / x[-1] for x in spacial_pattern[:(len(X) - 1)//2]])
    logging.debug(f"Total distance: {total_distance}")
    interpolated_reading = 0
    # print(len(X))
    # print(len(spacial_pattern))
    # exit()
    for i,feature in enumerate(X):
        w = (1 / spacial_pattern[i][-1]) / total_distance
        interpolated_reading += w * feature
    logging.debug(f"interpolated_reading: {interpolated_reading}")
    
    level = levelize(interpolated_reading, bins)
    logging.debug(f"level: {level}")
    return level, spacial_pattern

def gaussianSCG_interpolation(X, spacial_pattern, bins, avg_distance_cache = None):
    if spacial_pattern is None:
        logging.debug(f"No spacial pattern so defaulting to avg")
        return linear_interpolation_no_spacial(X, bins), Rbf()
    X = X[:(len(X) - 1)//2]
    x_locs = [i[0] for i in spacial_pattern[:len(X)]]
    y_locs = [i[1] for i in spacial_pattern[:len(X)]]
    interpolator = Rbf(x_locs, y_locs, X, function = 'gaussian')
    interpolated_reading = interpolator(spacial_pattern[-1][0], spacial_pattern[-1][1])
    logging.debug(f"interpolated_reading: {interpolated_reading}")
    
    level = levelize(interpolated_reading, bins)
    logging.debug(f"level: {level}")
    return level, interpolator

def linear_interpolation_no_spacial(X, bins):
    # First N features are current spacial
    # the last N+1 are temporal from the last period.
    # If locations is None, we have no location data
    # and best we can do is an average.
    logging.debug("Calling")
    logging.debug(X)  
    X = list(X)
    logging.debug(X)

    N = (len(X) - 1) // 2
    logging.debug(N)
    # spacial_readings = X[:N]
    spacial_readings = X[:-1:2]
    logging.debug(spacial_readings)
    logging.debug(sum(spacial_readings) / N)
    level = levelize(sum(spacial_readings) / N, bins)
    logging.debug(level)
    return level, level

class SimpleBaseline:
    def __init__(self, classifer_type, x_locs, y_loc, spacial_pattern, bins):
        self.classifer_type = classifer_type
        self.x_locs = x_locs
        self.y_loc = y_loc
        self.spacial_pattern = spacial_pattern
        
        self.avg_distance_cache = None
        self.bins = bins
        self.system_stats = systemStats()
        self.system_stats.state_control_log.append(
                [0, 0, None]
            )
        self.system_stats.state_control_log_altered.append(
                [0, 0, None]
            )
        self.system_stats.last_seen_window_length = 1000

        self.random_state = None
        self.poisson = 6
        self._random_state = check_random_state(self.random_state)
        self.cancelled = False
        self.ex = -1
        
        self.classes = None
        self._train_weight_seen_by_model = 0
        
        self.num_states = 1

        self.last_seen_label = 0
        
    def reset(self):
        pass

    def add_eval(self, X, y, prediction, correctly_classifies, ex):
        self.system_stats.add_prediction(
                X,
                y,
                prediction,
                correctly_classifies,
                ex
            )

    def partial_fit(self, X, y, classes=None, sample_weight=None, masked = False):
        """
        Fit an array of observations.
        Splits input into individual observations and
        passes to a helper function _partial_fit.
        Randomly weights observations depending on 
        Config.
        """
        # print(f"passing, X: {X}, y: {y}, m: {masked}")
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError(
                    'Inconsistent number of instances ({}) and weights ({}).'
                    .format(row_cnt, len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    self.ex += 1
                    if self.poisson >= 1:
                        k = self._random_state.poisson(self.poisson)
                        sample_weight[i] = k
                    self._partial_fit(X[i], y[i], sample_weight[i], masked)

    def get_inferrence(self, X):
        row_cnt, _ = get_dimensions(X)
        return_inferrence = []
        if self.classifer_type == 'linear':
            inferrence_func = linear_interpolation
        if self.classifer_type == 'normSCG':
            inferrence_func = gaussianSCG_interpolation
        if self.classifer_type == 'temporal':
            inferrence_func = temporal_interpolation
        if self.classifer_type == 'OK':
            inferrence_func = OK_interpolation
            
        for i in range(row_cnt):
            if self.avg_distance_cache is None:
                distances = []
                for s1 in self.spacial_pattern[:(len(X[i]) - 1)//2 - 1]:
                    for s2 in self.spacial_pattern[:(len(X[i]) - 1)//2 - 1]:
                        if s1 == s2:
                            continue
                        s1_x = s1[0]
                        s1_y = s1[1]
                        s2_x = s2[0]
                        s2_y = s2[1]
                        x_delta = s1_x - s2_x
                        y_delta = s1_y - s2_y
                        distance = math.sqrt(math.pow(x_delta, 2) + math.pow(y_delta, 2))
                        distances.append(distance)
                avg_distance = sum(distances) / len(distances)
                max_distance = max(distances)
                self.avg_distance_cache = avg_distance
            interpolated_value, classifier = inferrence_func(np.concatenate([X[i], self.last_seen_label], axis = None), self.spacial_pattern, self.bins, self.avg_distance_cache)
            self.classifier = classifier
            return_inferrence.append(interpolated_value)
        # print(f"prediction: {return_inferrence}")
        return return_inferrence

    def get_imputed_label(self, X):
        return self.last_seen_label

    def _partial_fit(self, X, y, sample_weight, masked = False):
        """
        Partially fit on a single observation.
        """
        # print(f"X: {X}, y: {y}, m: {masked}")
        # Predict before we fit, to detect drifts.
        prediction = self.get_inferrence([X])
        label = y if not masked else self.get_imputed_label(X)
        correctly_classifies = prediction == label
        # print(f"masked: {masked}")
        # print(f"Settings last to: {label}")
        self.last_seen_label = label

        self.system_stats.add_prediction(
                X,
                label,
                prediction,
                correctly_classifies,
                self.ex
            )

    def predict(self, X, last_seen_label = None):
        """
        Predict using the model of the currently active state.
        """
        return self.get_inferrence(X)

    def reset_stats(self):
        """
        Reset logs of states, (Call after they have been writen)
        """
        self.system_stats.model_stats.sliding_window_accuracy_log = []
        self.system_stats.correct_log = []
        self.system_stats.p_log = []
        self.system_stats.y_log = []

    def finish_up(self, ex):
        self.system_stats.state_control_log[-1][2] = ex
        self.system_stats.state_control_log_altered[-1][2] = ex
        return []


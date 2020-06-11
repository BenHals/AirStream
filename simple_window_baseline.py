from copy import deepcopy
import time
import numpy as np
import sys
import math
import logging
import scipy.stats
from skmultiflow.utils import get_dimensions, check_random_state
from skmultiflow.meta import AdaptiveRandomForest
from systemStats import systemStats
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

gaussian_cache = {}
def get_size(obj, seen=None):
        """Recursively finds size of objects"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        return size
def levelize(v):
    # return np.digitize(v, [50, 100, 150, 200])
    return np.digitize(v, [12, 35.5, 55.5, 150.5, 250.5])

def temporal_interpolation(X, spacial_pattern):
    level = levelize(X[-1])
    logging.debug(f"level: {level}")
    return level
def linear_interpolation(X, spacial_pattern):
    if spacial_pattern is None:
        logging.debug(f"No spacial pattern so defaulting to avg")
        return linear_interpolation_no_spacial(X)
    
    print([x[-1] for x in spacial_pattern])
    total_distance = sum([1 / x[-1] for x in spacial_pattern])
    logging.debug(f"Total distance: {total_distance}")
    interpolated_reading = 0
    for i,feature in enumerate(X):
        w = (1 / spacial_pattern[i][-1]) / total_distance
        interpolated_reading += w * feature
    logging.debug(f"interpolated_reading: {interpolated_reading}")
    
    level = levelize(interpolated_reading)
    logging.debug(f"level: {level}")
    return level

def gaussian_interpolation(X, spacial_pattern):
    if spacial_pattern is None:
        logging.debug(f"No spacial pattern so defaulting to avg")
        return linear_interpolation_no_spacial(X)
    
    total_distance = sum([x[-1] for x in spacial_pattern])
    avg_distance = total_distance / len(spacial_pattern)
    if avg_distance in gaussian_cache:
        distribution = scipy.stats.norm(0, avg_distance)
    else:
        distribution = gaussian_cache[avg_distance]
    logging.debug(f"Total distance: {total_distance}")
    interpolated_reading = 0
    weights = distribution.pdf([spacial_pattern[i][-1] for i,feature in enumerate(X)])
    for i,feature in enumerate(X):
        # w = scipy.stats.norm.pdf(spacial_pattern[i][-1], 0, avg_distance)
        # w = distribution.pdf(spacial_pattern[i][-1])
        w = weights[i]
        interpolated_reading += w * feature
    logging.debug(f"interpolated_reading: {interpolated_reading}")
    
    level = levelize(interpolated_reading)
    logging.debug(f"level: {level}")
    return level

def linear_interpolation_no_spacial(X):
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
    level = levelize(sum(spacial_readings) / N)
    logging.debug(level)
    return level

class SimpleWindowBaseline:
    def __init__(self, classifer_type, x_locs, y_loc, spacial_pattern):
        self.classifer_type = classifer_type
        self.t = 0
        self.window_X = []
        self.window_x_GP = []
        self.window_y = []
        self.window_y_GP = []
        self.x_locs = x_locs
        self.y_loc = y_loc
        self.spacial_pattern = spacial_pattern
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
        self.classifier = None

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
        if self.classifier is None:
            if self.classifer_type == 'tree':
                self.classifier = DecisionTreeClassifier()
                self.classifier.fit(self.window_X, self.window_y)
                # print(self.window_X)
                # print(self.window_y)
                print(self.classifier.feature_importances_)
                print("trained")
                self.size = get_size(self.classifier) + get_size(self.window_X) + get_size(self.window_y)
                # self.size = get_size(self.classifier.tree_)
                print(self.size)
            if self.classifer_type == 'tree9':
                self.classifier = DecisionTreeClassifier()
                self.classifier.fit(self.window_X, self.window_y)
                # print(self.window_X)
                # print(self.window_y)
                print(self.classifier.feature_importances_)
                print("trained")
                self.size = get_size(self.classifier) + get_size(self.window_X) + get_size(self.window_y)
                # self.size = get_size(self.classifier.tree_)
                print(self.size)
            if self.classifer_type == 'OK':
                self.classifier = GaussianProcessClassifier(copy_X_train=False)
                print(self.window_x_GP)
                train_X = np.vstack(self.window_x_GP)
                print(train_X.shape)
                print(train_X)
                print(len(self.window_y))
                self.classifier.fit(train_X, self.window_y_GP)
                # print(self.classifier.feature_importances_)
                # exit()
        
        if len(self.window_X) > 0:
            for i in range(row_cnt):
                test_X = np.concatenate([X[i], self.last_seen_label], axis = None).reshape(1, -1)
                if self.classifer_type == 'OK':
                    t_x, t_y, t_t, t_d = self.spacial_pattern[-1]
                    test_X = np.array([t_x, t_y, self.t]).reshape(1, -1)
                return_inferrence.append(self.classifier.predict(test_X))
        else:
            return_inferrence.append(0)
        return return_inferrence

    def get_imputed_label(self, X):
        return self.last_seen_label
        
    def _partial_fit(self, X, y, sample_weight, masked = False):
        """
        Partially fit on a single observation.
        """
        # print(f"X: {X}, y: {y}, m: {masked}")
        # Predict before we fit, to detect drifts.
        prediction = self.get_inferrence([X]) if not self.classifier is None else self.last_seen_label
        label = y if not masked else self.get_imputed_label(X)
        self.window_X.append(np.concatenate([X, self.last_seen_label], axis = None))
        for i,x in enumerate(X):
            if self.spacial_pattern is None:
                continue
            x_x = self.spacial_pattern[i][0]
            x_y = self.spacial_pattern[i][1]
            x_t = self.spacial_pattern[i][2]
            if x_t > 1:
                break
            self.window_x_GP.append(np.array([x_x, x_y, self.t]))
            self.window_y_GP.append(levelize(x))
        self.t += 1
        self.window_y.append(label)
        if not masked:
            self.last_seen_label = label
        # prediction = self.get_inferrence([X])
        # correctly_classifies = prediction == y

        # self.system_stats.add_prediction(
        #         X,
        #         y,
        #         prediction,
        #         correctly_classifies,
        #         self.ex
        #     )

    def predict(self, X, last_seen_label = None):
        """
        Predict using the model of the currently active state.
        """
        return self.get_inferrence(X)[0]

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


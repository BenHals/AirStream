from copy import deepcopy
import time
import numpy as np
import math
import logging
import scipy.stats
import itertools

from skmultiflow.utils import get_dimensions, check_random_state
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.utils import normalize_values_in_dict
from skikaprivate.classifiers.FSMClassifier.fsm.systemStats import systemStats

def updated_predict_proba(self, X):
        """ Predicts probabilities of all label of the X instance(s)
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.
        Returns
        -------
        numpy.array
            Predicted the probabilities of all the labels for all instances in X.
        """
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = deepcopy(self.get_votes_for_instance(X[i]))
            if votes == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append([0])
            else:
                if sum(votes.values()) != 0:
                    votes = normalize_values_in_dict(votes, inplace=False)
                if self.classes is not None:
                    y_proba = np.zeros(int(max(self.classes)) + 1)
                else:
                    y_proba = np.zeros(int(max(votes.keys())) + 1)
                for key, value in votes.items():
                    y_proba[int(key)] = value
                predictions.append(y_proba)
        # Set result as np.array
        if self.classes is not None:
            predictions = np.asarray(predictions)
        else:
            # Fill missing values related to unobserved classes to ensure we get a 2D array
            predictions = np.asarray(list(itertools.zip_longest(*predictions, fillvalue=0.0))).T
        return predictions

HoeffdingTree.predict_proba = updated_predict_proba

gaussian_cache = {}
def levelize(v, bins):
    # return np.digitize(v, [50, 100, 150, 200])
    # return np.digitize(v, [12, 35.5, 55.5, 150.5, 250.5])
    b = bins if not bins is None else [12, 35.5, 55.5, 150.5, 250.5]
    leveled = np.digitize(v, b)
    return leveled

def get_arf_active_state(classifier):
    state = max([x.nb_drifts_detected for x in classifier.ensemble])
    return state

class SimpleBaselineSKMF:
    def __init__(self, classifer_type, x_locs, y_loc, spacial_pattern, bins):
        self.classifer_type = classifer_type
        if classifer_type == 'arf':
            self.classifier = AdaptiveRandomForest()

        self.x_locs = x_locs
        self.y_loc = y_loc
        self.spacial_pattern = spacial_pattern
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

        self.last_label = 0

        self.active_state = 0
        
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


    def get_active_state(self):
        if self.classifer_type == 'arf':
            return get_arf_active_state(self.classifier)


    def _partial_fit(self, X, y, sample_weight, masked = False):
        """
        Partially fit on a single observation.
        """
        # print(f"X: {X}, y: {y}, m: {masked}")
        # Predict before we fit, to detect drifts.
        prediction = self.classifier.predict(np.concatenate([X, self.last_label], axis = None).reshape(1, -1))
        label = y if not masked else prediction[0]
        # print(np.concatenate([X, self.last_label], axis = None).reshape(1, -1))
        # print([label])
        self.classifier.partial_fit(np.concatenate([X, self.last_label], axis = None).reshape(1, -1), [label])
        self.last_label = label
        self.active_state = self.get_active_state()
        correctly_classifies = prediction == label

        self.system_stats.add_prediction(
                X,
                label,
                prediction,
                correctly_classifies,
                self.ex
            )

    def predict(self, X, last_label = None):
        """
        Predict using the model of the currently active state.
        """
        return self.classifier.predict(np.concatenate([X, self.last_label], axis = None).reshape(1, -1))

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


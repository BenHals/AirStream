import math
import logging
import time
from collections import Counter, deque
from copy import deepcopy
from skmultiflow.trees.hoeffding_tree import HoeffdingTree

from skmultiflow.utils import get_dimensions, check_random_state
from skmultiflow.drift_detection.adwin import ADWIN
from systemStats import systemStats
import numpy as np
import scipy.stats


def make_detector(warn=False, s=1e-5):
    sensitivity = s * 2 if warn else s
    return ADWIN(delta=sensitivity)


class state:
    def __init__(self, id, learner):
        self.id = id
        self.classifier = learner
        self.seen = 0
    
    def __str__(self):
        return f"<State {self.id}>"
    
    def __repr__(self):
        return self.__str__()

class DSClassifier:
    def __init__(self, 
                suppress=False,
                concept_limit=-1,
                memory_management='rA',
                learner=None,
                window=100,
                sensitivity=0.05,
                concept_chain=None,
                optimal_selection=False,
                optimal_drift=False,
                rand_weights=False,
                poisson=10,
                similarity_measure='KT',
                merge_strategy="both",
                merge_similarity=0.9,
                allow_backtrack=False,
                allow_proactive_sensitivity=False,
                num_alternative_states=5,
                conf_sensitivity_drift=0.05,
                conf_sensitivity_sustain=0.125,
                min_proactive_stdev=500,
                alt_test_length=2000,
                alt_test_period=2000,
                max_sensitivity_multiplier=1.5):

        if learner is None:
            raise ValueError('Need a learner')
        
        # concept_limit is the maximum number of concepts
        # which can be stored in the repository
        self.concept_limit = concept_limit

        # memory_management is the memory management
        # strategy used.
        self.memory_management = memory_management

        # learner is the classifier used by each state.
        # papers use HoeffdingTree from scikit-multiflow
        self.learner = learner

        # window_min is the minimum size of window used
        # when a concept drift is detected to select the
        # new concept.
        self.window_min = window

        # window is dynamic based on the two detectors
        self.window = window

        # sensitivity is the sensitivity of the concept
        # drift detector
        self.sensitivity = sensitivity
        self.base_sensitivity = sensitivity
        self.max_sensitivity = sensitivity * max_sensitivity_multiplier
        self.TEST_sensitivity = []
        self.current_sensitivity = sensitivity

        # suppress debug info
        self.suppress = suppress

        # optimal knowledge of drift in the stream.
        # set to None if not known (default)
        self.concept_chain = concept_chain
        self.optimal_selection = optimal_selection
        self.optimal_drift = optimal_drift

        # rand_weights is if a strategy is setting sample
        # weights for training
        self.rand_weights = poisson > 1

        # poisson is the strength of sample weighting
        # based on leverage bagging
        self.poisson = poisson

        # merge_similarity is the distance at which similar
        # concepts are merged
        self.merge_similarity = merge_similarity

        # allow_proactive_sensitivity allows proactive sensitivity
        self.allow_proactive_sensitivity = allow_proactive_sensitivity

        # num_alternative_states is the K parameter controlling
        # how many inactive states are considered
        self.num_alternative_states = num_alternative_states

        # the thresholds for quick and sustained repair detection
        # respectively
        self.conf_sensitivity_drift = conf_sensitivity_drift
        self.conf_sensitivity_sustain = conf_sensitivity_sustain
        self.min_proactive_stdev = min_proactive_stdev

        # length and period of repair testing
        self.alt_test_length = alt_test_length
        self.alt_test_period = alt_test_period

        # setup initial drift detectors
        self.detector = make_detector(s = sensitivity)
        self.warn_detector = make_detector(s = self.get_warning_sensitivity(sensitivity))
        self.in_warning = False
        self.last_warning_point = 0

        # initialize waiting state. If we don't have enough
        # data to select the next concept, we wait until we do.
        self.waiting_for_concept_data = False

        # init the current number of states
        self.max_state_id = 0

        # init randomness
        self.random_state = None
        self._random_state = check_random_state(self.random_state)

        
        self.ex = -1
        self.classes = None
        self._train_weight_seen_by_model = 0

        # set the similarity measure to determine the similarity
        # of repository concepts to current stream.
        self.similarity_measure = similarity_measure
        
        # init backtracking data
        self.restore_state = None
        self.restore_state_set_point = 0
        self.restore_point_type = None
        self.allow_backtrack = allow_backtrack
        self.active_state_is_new = True

        # init data which is exposed to evaluators 
        self.found_change = False
        self.num_states = 1
        self.active_state = self.max_state_id
        self.alternative_states = []
        self.reset_alternative_states = False
        self.states = []
        self.set_restore_state = None
        self.load_restore_state = None
        self.signal_difference_backtrack = False
        self.signal_confidence_backtrack = False
        self.alternative_states_difference_confidence = {}

        # track the last predicted label
        self.last_label = 0

        # set up repository
        self.state_repository = {}
        self.testing_state_repository = {}
        self.testing_state_stats = {}
        self.testing_state_main_comparison = {'perf_window': deque(), 'perf_sum': 0}
        init_id = self.max_state_id
        self.max_state_id += 1
        init_state = state(init_id, self.learner())
        self.state_repository[init_id] = init_state
        self.active_state_id = init_id

        # set up performance history
        self.recent_accuracy = []
        self.recent_non_masked_history = deque()
        self.recent_non_masked_history_sum = 0
        self.history = []
        self.testing_state_history = []
        self.testing_state_performance = {}

        self.inactive_test_min = 100
        self.inactive_test_grace = 200


    def get_warning_sensitivity(self, s):
        return s * 2

    def get_active_state(self):
        return self.state_repository[self.active_state_id]


    def make_state(self):
        new_id = self.max_state_id
        self.max_state_id += 1
        return new_id, state(new_id, self.learner())


    def reset(self):
        pass
    

    def get_temporal_x(self, X):
        # return np.concatenate([X, self.last_label], axis = None)
        return np.concatenate([X, 0], axis = None)

    def predict(self, X):
        """
        Predict using the model of the currently active state.
        """
        logging.debug("Predicting")
        # ll = self.last_label if last_label is None else last_label
        temporal_X = self.get_temporal_x(X)
        logging.debug(f"temporal_X: {temporal_X}")

        # ll = 0
        return self.get_active_state().classifier.predict([temporal_X])

    def partial_fit(self, X, y, classes=None, sample_weight=None, masked = False):
        """
        Fit an array of observations.
        Splits input into individual observations and
        passes to a helper function _partial_fit.
        Randomly weights observations depending on 
        Config.
        """
        # print(f"Masekd in PF: {masked}")
        # print(f"passing, X: {X}, y: {y}, m: {masked}")
        if masked:
            return
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
                    if self.rand_weights and self.poisson >= 1:
                        # k = self._random_state.poisson(self.poisson)
                        k = self.poisson
                        sample_weight[i] = k
                    self._partial_fit(X[i], y[i], sample_weight[i], masked)

    def get_imputed_label(self, X, prediction, last_label):
        """ Get a label.
        Imputes when the true label is masked

        """
        return prediction



    def _partial_fit(self, X, y, sample_weight, masked = False):
        self.reset_alternative_states = False
        logging.debug(f"Partial fit on X: {X}, y:{y}, masked: {masked}, using state {self.active_state_id}")

        temporal_X = self.get_temporal_x(X)
        prediction = self.predict(temporal_X)[0]

        label = y if not masked else self.get_imputed_label(X=X, prediction = prediction, last_label = self.last_label)
        self.last_label = label

        # correctly_classified from the systems point of view.
        correctly_classifies = prediction == label

        # Save the information on the sample for performance calculations
        self.history.append(
            {
                "temporal_X": temporal_X,
                "label": label,
                "sample_weight": sample_weight,
                "p": prediction,
                "active_model": self.active_state_id,
                "masked": masked,
                "ex": self.ex

            }
        )

        # init defaults for trackers
        backtrack_target = None
        found_change = False
        current_sensitivity = self.get_current_sensitivity()

        # The following section uses the label, so should only run if 
        # we have good data, i.e. is not masked.
        if not masked:

            # maintain a recent history of good data
            # recent_non_masked_history is a window of observations
            # since the current active state was set.
            self.recent_non_masked_history.append(self.history[-1])
            self.recent_non_masked_history_sum += 1 if self.history[-1]['p'] == self.history[-1]['label'] else 0
            if len(self.recent_non_masked_history) > 500:
                removed_element = self.recent_non_masked_history.popleft()
                self.recent_non_masked_history_sum -= 1 if removed_element['p'] == removed_element['label'] else 0

            # recent_accuracy is a list of recent window accuracies over the seen
            # good data. Does not include the first 10 of these as the number of samples
            # is too small, and is from when the current active window was set.
            if len(self.recent_non_masked_history) > self.inactive_test_min - 10:
                # self.recent_accuracy.append(sum([1 if e['p'] == e['label'] else 0 for e in self.recent_non_masked_history]) / len(self.recent_non_masked_history))
                self.recent_accuracy.append(self.recent_non_masked_history_sum / len(self.recent_non_masked_history))

            # Fit the classifier of the current state.
            self.get_active_state().classifier.partial_fit(
                np.asarray([temporal_X]),
                np.asarray([label]),
                sample_weight=np.asarray([sample_weight])
            )
            self.get_active_state().seen += 1

            # if backtracking is allowed and
            # we are within alt_test_length of the repair testing starting
            # we test inactive states. This sets backtrack_target to be 
            # the best inactive state if we are confident it is better than the
            # main state, other wise None if the main state is best.
            if self.allow_backtrack:
                # Testing is expensive, so we only test for a length of time
                # after each restore point is placed.
                logging.debug(len(self.testing_state_repository))
                logging.debug(self.ex - self.restore_state_set_point)
                if (self.ex - self.restore_state_set_point) < self.alt_test_length:
                    logging.debug("testing")
                    backtrack_target = self.test_inactive_states(temporal_X, label, sample_weight, prediction)
                    logging.debug(self.testing_state_performance)
            else:
                backtrack_target = None


            # Add error to detectors
            self.detector.delta = current_sensitivity
            self.warn_detector.delta = self.get_warning_sensitivity(current_sensitivity)
            self.detector.add_element(int(correctly_classifies))
            self.warn_detector.add_element(int(correctly_classifies))

            # If the warning detector fires we record the position
            # and reset. We take the most recent warning as the
            # start of out window, assuming this warning period
            # contains elements of the new state.
            if self.warn_detector.detected_change():
                self.in_warning = True
                self.last_warning_point = self.ex
                self.warn_detector = make_detector(s = self.get_warning_sensitivity(current_sensitivity))
            
            if not self.in_warning:
                self.last_warning_point = max(0, self.ex - 100)
            self.window = max(self.window_min, self.ex - self.last_warning_point)

            # If the main state trigers, or we held off on changing due to lack of data,
            # trigger a change
            found_change = self.detector.detected_change() or self.waiting_for_concept_data
        
        if found_change:
            logging.debug("Found Change")
            self.in_warning = False

            # Find the inactive models most suitable for the current stream. Also return a shadow model 
            # trained on the warning period.
            # If none of these have high accuracy, hold off the adaptation until we have the data.
            ranked_alternatives, use_shadow, shadow_model, can_find_good_model = self.rank_inactive_models_for_suitability()
            logging.debug(f"Candidates: {ranked_alternatives}")
            if can_find_good_model:

                # We may want to backtrack the active state to make sure it did not ingest
                # samples from a different concept.
                # If we have a restore state of the active state, and we are not transitioning to
                # the same state, we check how many observations would be deleted by this.
                # if this is less than some proportion (a third here) we take the tradeoff and backtrack
                # to the restore state model.
                restore_state = self.restore_state
                if restore_state and restore_state.id == self.active_state_id and ((len(ranked_alternatives) > 0 and ranked_alternatives[-1] != self.active_state_id) or use_shadow):
                    logging.debug(f"Restore point set at {self.restore_state_set_point}")
                    time_delta = self.ex - self.restore_state_set_point
                    logging.debug(f"{time_delta}")
                    proportion_deleted_by_backtrack = time_delta / self.get_active_state().seen
                    logging.debug(f"{proportion_deleted_by_backtrack}")
                    if proportion_deleted_by_backtrack < 0.33:
                        logging.debug(f"Restoring {restore_state.id}")
                        self.state_repository[restore_state.id] = deepcopy(self.restore_state)

                # If we determined the shadow is the best model, we mark it as a new state,
                # copy the model trained on the warning period across and set as the active state
                if use_shadow:
                    logging.debug(f"Transition to shadow")
                    self.active_state_is_new = True
                    shadow_id, shadow_state = self.make_state()
                    shadow_state.classifier = shadow_model
                    self.state_repository[shadow_id] = shadow_state
                    self.active_state_id = shadow_id
                else:
                    # Otherwise we just set the found state to be active
                    logging.debug(f"Transition to {ranked_alternatives[-1]}")
                    self.active_state_is_new = False
                    transition_target_id = ranked_alternatives[-1]
                    self.active_state_id = transition_target_id

                # We reset drift detection as the new performance will not 
                # be the same, and reset history for the new state.
                self.waiting_for_concept_data = False
                self.detector = make_detector(s = current_sensitivity)
                self.warn_detector = make_detector(s = self.get_warning_sensitivity(current_sensitivity))
                self.recent_non_masked_history = deque()
                self.recent_non_masked_history_sum = 0
                self.recent_accuracy = []

                # Set the restore point to test for false positive drift
                self.place_restorepoint(ranked_alternatives, self.learner(), 'transition')

                # Test the newly made inactive states.
                self.test_inactive_states(temporal_X, label, sample_weight, prediction)
            else:
                # If we did not have enough data to find any good concepts to 
                # transition to, wait until we do.
                self.waiting_for_concept_data = True

        elif backtrack_target is not None:
            # We look for a backtrack if no change is detected.
            # First, we reset the active state model to its saved state.
            restore_state = self.restore_state
            logging.debug(f"Restoring {restore_state.id}")
            self.state_repository[restore_state.id] = deepcopy(self.restore_state)

            # If the backtrack is to a new model, we make the new state for it
            # otherwise we set it to be the indicated stored model.
            backtrack_to_new_model = backtrack_target == -1
            if backtrack_to_new_model:
                self.active_state_is_new = True
                new_id, target_state = self.make_state()
                target_state.classifier = self.testing_state_repository[-1].classifier
                self.state_repository[new_id] = target_state
            else:
                self.active_state_is_new = False
                target_state = self.state_repository[backtrack_target]
            self.active_state_id = target_state.id

            # We reset data, including testing state performance
            self.detector = make_detector(s = current_sensitivity)
            self.warn_detector = make_detector(s = self.get_warning_sensitivity(current_sensitivity))
            self.recent_non_masked_history = deque()
            self.recent_non_masked_history_sum = 0
            self.recent_accuracy = []
            self.testing_state_history = []
            self.testing_state_repository = {}
            self.testing_state_stats = {}
            self.testing_state_main_comparison = {'perf_window': deque(), 'perf_sum': 0}
            self.reset_alternative_states = True

            # We place a new restore point by first finding models which are a good fit.
            ranked_alternatives, use_shadow, shadow_model, can_find_good_model = self.rank_inactive_models_for_suitability()
            if can_find_good_model:
                self.place_restorepoint(ranked_alternatives, shadow_model, 'transition')
                self.test_inactive_states(temporal_X, label, sample_weight, prediction)
            self.testing_state_history = []
        elif self.allow_backtrack:
            # If there is no change, or backtrack, we test to see if a periodic restore point test
            # should begin.
            should_place_restore_point = self.should_place_restore_point()
            if should_place_restore_point:
                self.window = self.window_min
                ranked_alternatives, use_shadow, shadow_model, can_find_good_model = self.rank_inactive_models_for_suitability()
                if can_find_good_model:
                    self.place_restorepoint(ranked_alternatives, shadow_model, 'transition')
                    self.test_inactive_states(temporal_X, label, sample_weight, prediction)



        # Set exposed info for evaluation.
        self.active_state = self.active_state_id
        self.found_change = found_change
        self.states = self.state_repository
        self.current_sensitivity = current_sensitivity
        self.alternative_states = [x for x in self.testing_state_performance.values()]




    
    def rank_inactive_models_for_suitability(self):
        t = 0.85
        shadow_t = 0.95
        shadow_m = 1.025

        logging.debug(f"Ranking inactive models on current stream, using last {self.window} elements.")
        recent_window = self.history[-self.window:]
        # Don't test on masked elements, we cannot rely on labels
        recent_window = [x for x in recent_window if not x["masked"]]
        if len(recent_window) == 0:
            return [[], False, None, False]
        recent_window_observations = set([e['ex'] for e in recent_window])

        recent_X = np.vstack([x["temporal_X"] for x in recent_window])
        logging.debug(f"Recent features have shape {recent_X.shape}")
        recent_labels = np.array([x["label"] for x in recent_window])

        shadow_model = self.learner()
        shadow_seen = 0
        shadow_predictions = []
        for X, y in zip(reversed([e['temporal_X'] for e in recent_window]), reversed([e['label'] for e in recent_window])):
            prediction = shadow_model.predict([X])[0]

            # Throw away initial predictions while system is learning
            if shadow_seen > 10:
                shadow_predictions.append(prediction)
            shadow_model.partial_fit([X], [y])
            shadow_seen += 1
        if len(shadow_predictions) < 15:
            return [[], False, None, False]
        shadow_predictions.reverse()
        shad_acc, shad_k_t, shad_k_m, shad_k_s = get_stats(list(zip(shadow_predictions, recent_labels)))
        logging.debug(f"Shadow performance: Acc {shad_acc}, kt: {shad_k_t}, km: {shad_k_m}, ks: {shad_k_s}")

        state_performace_by_id = []
        filter_states = set()
        for state_id in self.state_repository:
            state = self.state_repository[state_id]
            state_predictions = state.classifier.predict(recent_X)

            # States can be biased if they were the active state that trained on an element.
            # In that case we look to see what that model initially predicted.
            unbiased_predictions = []
            for i,recent_element in enumerate(recent_window):
                if recent_element["active_model"] == state_id:
                    unbiased_predictions.append(recent_element["p"])
                else:
                    unbiased_predictions.append(state_predictions[i])
            logging.debug(f"State {state_id} predictions have length {len(unbiased_predictions)}")

            acc, kt, km, ks = get_stats(list(zip(unbiased_predictions, recent_labels)))
            logging.debug(f"State {state_id} performance: Acc {acc}, kt: {kt}, km: {km}, ks: {ks}")

            # We test to see if the accuracy on the current stream is similar to past uses
            # of a state. If not, we do not consider it.
            # We don't want to retest elements already tested in recent_window.
            state_predictions = [e for e in self.history if e["active_model"] == state_id and not e["masked"] and e['ex'] not in recent_window_observations]
            state_predictions = state_predictions[-round(len(state_predictions) / 3):]
            recent_acc, recent_kt, recent_km, recent_ks = get_stats([(e['p'], e['label']) for e in state_predictions])
            logging.debug(f"Recent performance: Acc {recent_acc}, kt: {recent_kt}, km: {recent_km}, ks: {recent_ks}")

            # Active state had a change detection, so we penalize it.
            active_state_penalty = 1.05 if state_id == self.active_state_id else 1
            state_similarity_to_shadow = get_kappa_agreement(shadow_predictions, unbiased_predictions[-len(shadow_predictions):])

            # We test if kt on the current stream is above a threshold of kt on last use.
            # We also test if the current model has some similarity to the new shadow model.
            # If neither, we filter
            if kt * active_state_penalty <= recent_kt * t and state_similarity_to_shadow * active_state_penalty <= shadow_t:
                filter_states.add(state_id)
                logging.debug(f"State {state_id} filtered")
            else:
                state_performace_by_id.append((state_id, kt))

        # State suitability in sorted (ascending) order
        state_performace_by_id.sort(key = lambda x: x[1])

        use_shadow = True
        if len(state_performace_by_id) > 0:

            if state_performace_by_id[-1][1] > shad_k_t * shadow_m:
                use_shadow = False
                logging.debug("Top state has higher kt than shadow threshold")
            else:
                current_system_acc, current_system_kt, current_system_km, current_system_ks = get_stats([(e['p'], e['label']) for e in self.history[-2000:] if not e['masked']])
                growth_potential = 1 - current_system_kt
                if state_performace_by_id[-1][1] > max(current_system_kt + growth_potential * 0.5, 0.8):
                    use_shadow = False
                    logging.debug("Top state high performance")

        # shadow_model = self.learner()
        # shadow_model = HoeffdingTree()
        return [[x[0] for x in state_performace_by_id], use_shadow, shadow_model, True]


    def place_restorepoint(self, model_ids_to_test, shadow_model, restore_point_type):
        if not self.allow_backtrack:
            return
        # Take a copy of the current state
        logging.debug("Placing Restore point")
        logging.debug(f"Ranked Alternatives: {model_ids_to_test}")
        self.restore_state = deepcopy(self.get_active_state())
        self.restore_state_set_point = self.ex
        self.restore_point_type = restore_point_type

        self.testing_state_history = []
        self.testing_state_repository = {}
        self.testing_state_stats = {}
        self.testing_state_main_comparison = {'perf_window': deque(), 'perf_sum': 0}
        self.reset_alternative_states = True

        if len(model_ids_to_test) > 0:
            for state_id in model_ids_to_test[-self.num_alternative_states:]:
                if state_id == self.active_state_id:
                    continue
                self.testing_state_repository[state_id] = deepcopy(self.state_repository[state_id])
                self.testing_state_stats[state_id] = {'perf': [], 'sustained_confidence_sum': 0, 'quick_change_detector': make_detector(s = self.conf_sensitivity_drift), "found_quick_change": False, 'seen': 0, 'perf_window': deque(), 'perf_sum': 0}
        
        # if not self.active_state_is_new:
        self.testing_state_repository[-1] = state(-1, deepcopy(shadow_model))
        self.testing_state_stats[-1] = {'perf': [],'sustained_confidence_sum': 0, 'quick_change_detector': make_detector(s = self.conf_sensitivity_drift), "found_quick_change": False, 'seen': 0, 'perf_window': deque(), 'perf_sum': 0}
        logging.debug(self.testing_state_repository)
        logging.debug(self.testing_state_stats)
        


    def should_place_restore_point(self):
        """ Testing strategies could involve looking for local maxima,
        i.e. restoring to a known good point. For not just periodic.
        """
        # Check time since last restore point
        if not self.allow_backtrack:
            return False
        above_wait_period = (self.ex - self.restore_state_set_point) > self.alt_test_period

        return above_wait_period

    def test_inactive_states(self, temporal_X, label, sample_weight, p):
        """ Given the current observation, we test the inactive states in the 
        testing repository. This repository contains copies of the K most similar
        states to the current stream.

        """
        
        # We shouldn't do anything is testing is not allowed.
        if not self.allow_backtrack:
            return

        backtrack_states = []
        self.testing_state_performance = {}

        is_correct = 1 if p == label else 0
        self.testing_state_main_comparison['perf_window'].append(is_correct)
        self.testing_state_main_comparison['perf_sum'] += is_correct

        # for each state in the testing repository, we get the prediction it would
        # have made for the current observation. This is added to a history of inactive
        # state performance tagged with its ID.
        # We compare the performance of the state to the active state for all observations
        # since testing started.
        # We consider a normal distribution around the main state performance, and perform 
        # a t-test to determine the likelyhood of the inactive states performance being the
        # same or lower than the main state. We collect the p-value, if the average p-value
        # drops below 0.05 we have evidence that the inactive state is drawn from a higher
        # distribution and initiate a backtrack.
        # We also perform drift detection on the difference in performance, a sudden change
        # here when the performance of the inactive state is higher and increasing indicates
        # it is better suited so a quick change backtrack is initiated.
        for inactive_state_id in self.testing_state_repository:
            logging.debug(f"Testing alt state {inactive_state_id}")
            inactive_state = self.testing_state_repository[inactive_state_id]
            state_prediction = inactive_state.classifier.predict([temporal_X])[0]
            logging.debug(f"Predicted {state_prediction}")
            self.testing_state_history.append(
                {
                    "temporal_X": temporal_X,
                    "label": label,
                    "sample_weight": sample_weight,
                    "p": state_prediction,
                    "main_state_p": p,
                    "active_model": inactive_state_id,
                    "masked": False,
                    "ex": self.ex
                }
            )
            is_correct = 1 if state_prediction == label else 0
            self.testing_state_stats[inactive_state_id]['perf_window'].append(is_correct)
            inactive_state_seen = len(self.testing_state_stats[inactive_state_id]['perf_window'])
            self.testing_state_stats[inactive_state_id]['perf_sum'] += is_correct

            # Currently consider whole history, a sliding window might be better.
            # test_history = [e for e in self.testing_state_history if e['active_model'] == inactive_state_id and not e['masked']]
            # logging.debug(f"Seen {len(test_history)} samples")
            logging.debug(f"Seen {inactive_state_seen} samples")
            # state_accuracy = sum([1 if e['p'] == e['label'] else 0 for e in test_history]) / len(test_history)
            state_accuracy = self.testing_state_stats[inactive_state_id]['perf_sum'] / inactive_state_seen
            self.testing_state_stats[inactive_state_id]['perf'].append(state_accuracy)
            # main_state_accuracy = sum([1 if e['main_state_p'] == e['label'] else 0 for e in test_history]) / len(test_history)
            main_state_accuracy = self.testing_state_main_comparison['perf_sum'] / inactive_state_seen
            # if len(test_history) < self.inactive_test_min:
            if inactive_state_seen < self.inactive_test_min:
                self.testing_state_performance[inactive_state_id] = (inactive_state_id, state_prediction == label, inactive_state, state_accuracy, main_state_accuracy, len(self.testing_state_history))
                continue
            self.testing_state_performance[inactive_state_id] = (inactive_state_id, state_prediction == label, inactive_state, state_accuracy, main_state_accuracy, len(self.testing_state_history))

            # if the inactive state is the shadow state, we fit it.
            # We do not test against the shadow state if the active state
            # is too new, as this is just two unstable states against eachother.
            if inactive_state_id == -1:
                inactive_state.classifier.partial_fit(
                    np.asarray([temporal_X]),
                    np.asarray([label]),
                    sample_weight=np.asarray([sample_weight])
                )
                if self.get_active_state().seen < 750:
                    continue
            
            # Consider the kappa measure instead of the raw distance.
            # Since accuracy is capped at one, this scales the difference
            # by the distance to one, so a 5% increase at 90% is like a
            # 25% increase at 50%.
            kappa_measure = (state_accuracy - main_state_accuracy) / (1 - main_state_accuracy) if main_state_accuracy < 1 else 0

            logging.debug(self.recent_accuracy[-100:])
            logging.debug(np.std(self.recent_accuracy[-100:]))

            # Get the standard deviation as the max of the stdev of main state performance
            # and of inactive state performance, both scaled in the same way as the difference.
            # main_state_recent_acc_std = np.std(self.recent_accuracy[-100:])
            # test_state_recent_acc_std = np.std(self.testing_state_stats[inactive_state_id]['perf'])
            scaled_main_state_recent_acc_std = np.std(self.recent_accuracy[-100:]) / (1-main_state_accuracy) if main_state_accuracy < 1 else 0.0001
            scaled_test_state_recent_acc_std = np.std(self.testing_state_stats[inactive_state_id]['perf']) / (1-state_accuracy) if state_accuracy < 1 else 0.0001

            std_val = max(scaled_test_state_recent_acc_std, scaled_main_state_recent_acc_std)

            logging.debug(f"testing state acc: {state_accuracy}")
            logging.debug(f"main state acc: {main_state_accuracy}")
            # logging.debug(f"main state std: {main_state_recent_acc_std}")
            logging.debug(f"std val: {std_val}")


            # probability_drawn_from_main_state = (1 - scipy.stats.norm(loc = main_state_accuracy, scale = main_state_recent_acc_std).cdf(state_accuracy))
            # probability_drawn_from_main_state = (1 - scipy.stats.norm(loc = main_state_accuracy, scale = scaled_main_state_recent_acc_std).cdf(state_accuracy))

            # The probability is the inverse of the cdf of the normal distribution given by that stdev.
            probability_drawn_from_main_state = (1 - scipy.stats.norm(loc = 0, scale = std_val ).cdf(kappa_measure))
            logging.debug(f"probability_drawn_from_main_state: {probability_drawn_from_main_state}")

            # store the sum of the probability and number seen to calculate the average
            self.testing_state_stats[inactive_state_id]["sustained_confidence_sum"] += probability_drawn_from_main_state
            self.testing_state_stats[inactive_state_id]["seen"] += 1
            logging.debug(f"sustained_confidence_sum: {self.testing_state_stats[inactive_state_id]['sustained_confidence_sum']}")
            logging.debug(f"seen: {self.testing_state_stats[inactive_state_id]['seen']}")
            avg_sustained_confidence = self.testing_state_stats[inactive_state_id]["sustained_confidence_sum"] / self.testing_state_stats[inactive_state_id]["seen"]
            logging.debug(f"avg_sustained_confidence: {avg_sustained_confidence}")
            
            # Test the kappa with a drift detector
            self.testing_state_stats[inactive_state_id]["quick_change_detector"].add_element(state_accuracy - main_state_accuracy)
            self.testing_state_stats[inactive_state_id]["found_quick_change"] = (self.testing_state_stats[inactive_state_id]["quick_change_detector"].detected_change() and state_accuracy > main_state_accuracy) or self.testing_state_stats[inactive_state_id]["found_quick_change"]
            logging.debug(f"Quick change detected: {self.testing_state_stats[inactive_state_id]['found_quick_change']}")
            
            # Give a grace period for triggers as initial performance is unstable
            # if len(test_history) > self.inactive_test_grace:
            if inactive_state_seen > self.inactive_test_grace + self.inactive_test_min:
                t = 0.85
                
                if avg_sustained_confidence < self.conf_sensitivity_sustain:
                    logging.debug(f"Conf signal for state {inactive_state_id}")
                    min_sample_to_look_at = self.testing_state_history[0]['ex']
                    state_predictions = [e for e in self.history if e["active_model"] == inactive_state_id and not e["masked"] and e['ex'] > min_sample_to_look_at]
                    state_predictions = state_predictions[-round(len(state_predictions) / 3):]
                    recent_acc, recent_kt, recent_km, recent_ks = get_stats([(e['p'], e['label']) for e in state_predictions])
                    logging.debug(f"Recent performance: Acc {recent_acc}, kt: {recent_kt}, km: {recent_km}, ks: {recent_ks}")

                    # We test if kt on the current stream is above a threshold of kt on last use.
                    # We also test if the current model has some similarity to the new shadow model.
                    # If neither, we filter
                    if state_accuracy > recent_acc * t:
                        backtrack_states.append((inactive_state_id, state_accuracy))
                    else:
                        logging.debug(f"But acc was too different to normal so was dropped")
                if self.testing_state_stats[inactive_state_id]["found_quick_change"] and state_accuracy > main_state_accuracy:
                    logging.debug(f"Drift signal for state {inactive_state_id}")
                    min_sample_to_look_at = self.testing_state_history[0]['ex']
                    state_predictions = [e for e in self.history if e["active_model"] == inactive_state_id and not e["masked"] and e['ex'] > min_sample_to_look_at]
                    state_predictions = state_predictions[-round(len(state_predictions) / 3):]
                    recent_acc, recent_kt, recent_km, recent_ks = get_stats([(e['p'], e['label']) for e in state_predictions])
                    logging.debug(f"Recent performance: Acc {recent_acc}, kt: {recent_kt}, km: {recent_km}, ks: {recent_ks}")

                    # We test if kt on the current stream is above a threshold of kt on last use.
                    # We also test if the current model has some similarity to the new shadow model.
                    # If neither, we filter
                    if state_accuracy > recent_acc * t:
                        backtrack_states.append((inactive_state_id, state_accuracy))
                    else:
                        logging.debug(f"But acc was too different to normal so was dropped")
        
        # Return the backtrack state with highest accuracy
        if len(backtrack_states) > 0:
            logging.debug("Backtrack signalled")
            backtrack_states.sort(key = lambda x: x[1])
            logging.debug(backtrack_states)

            return backtrack_states[-1][0]

        return None
    def get_current_sensitivity(self):
        return self.base_sensitivity



def get_stats(results):
    predictions = [x[0] for x in results]
    if len(predictions) == 0:
        return 0, 0, 0, 0
    recent_y = [x[1] for x in results]
    # results = zip(predictions, recent_y)
    accuracy = sum(np.array(predictions) ==
                   np.array(recent_y)) / len(predictions)
    k_temporal_acc = 0
    k_majority_acc = 0
    gt_counts = Counter()
    our_counts = Counter()
    majority_guess = results[0][1]
    temporal_guess = results[0][1]
    for o in results:
        p = o[0]
        gt = o[1]
        if gt == temporal_guess:
            k_temporal_acc += 1
        if gt == majority_guess:
            k_majority_acc += 1
        gt_counts[gt] += 1
        our_counts[p] += 1

        majority_guess = gt if gt_counts[gt] > gt_counts[majority_guess] else majority_guess
        temporal_guess = gt
    k_temporal_acc = min(k_temporal_acc / len(results), 0.99999)
    k_temporal_acc = (accuracy - k_temporal_acc) / (1 - k_temporal_acc)
    k_majority_acc = min(k_majority_acc / len(results), 0.99999)

    k_majority_acc = (accuracy - k_majority_acc) / (1 - k_majority_acc)
    expected_accuracy = 0
    for cat in np.unique(predictions):
        expected_accuracy += min((gt_counts[cat] *
                                  our_counts[cat]) / len(results), 0.99999)
    expected_accuracy /= len(results)
    k_s = (accuracy - expected_accuracy) / (1 - expected_accuracy)

    return accuracy, k_temporal_acc, k_majority_acc, k_s


def get_kappa_agreement(A_preds, B_preds):
    A_counts = Counter()
    B_counts = Counter()
    similar = 0
    for pA, pB in zip(A_preds, B_preds):
        A_counts[pA] += 1
        B_counts[pB] += 1
        if pA == pB:
            similar += 1
    observed_acc = similar / len(A_preds)
    expected_sum = 0
    for cat in np.unique(pA + pB):
        expected_sum += min((A_counts[cat] *
                             B_counts[cat]) / len(A_preds), 0.99999)
    expected_acc = expected_sum / len(A_preds)
    k_s = (observed_acc - expected_acc) / (1 - expected_acc)
    return k_s
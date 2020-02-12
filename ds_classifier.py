import math
import logging
import time
from collections import Counter, deque
from copy import deepcopy

from skmultiflow.utils import get_dimensions, check_random_state
from skmultiflow.drift_detection.adwin import ADWIN
from skikaprivate.classifiers.FSMClassifier.fsm.fsm import FSM, deepcopy_fast
from skikaprivate.classifiers.FSMClassifier.fsm.systemStats import systemStats
import numpy as np
import scipy.stats


def make_detector(warn=False, s=1e-5):
    sensitivity = s * 10 if warn else s
    return ADWIN(delta=sensitivity)


class state:
    def __init__(self, id, learner):
        self.id = id
        self.classifier = learner

class DSClassifier:
    def __init__(self, 
                suppress=False,
                concept_limit=10,
                memory_management='rA',
                learner=None,
                window=50,
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
                conf_sensitivity_sustain=0.05,
                min_proactive_stdev=500,
                alt_test_length=2000,
                alt_test_period=2000,
                max_sensitivity_multiplier=1.5):

        if learner is None:
            raise ValueError('Need a learner')

        self.concept_limit = concept_limit
        self.memory_management = memory_management
        self.learner = learner
        self.window = window
        self.sensitivity = sensitivity

        self.base_sensitivity = sensitivity
        self.max_sensitivity = sensitivity * max_sensitivity_multiplier
        self.TEST_sensitivity = []
        self.current_sensitivity = sensitivity

        self.suppress = suppress
        self.concept_chain = concept_chain
        self.optimal_selection = optimal_selection
        self.optimal_drift = optimal_drift
        self.rand_weights = poisson > 1
        self.poisson = poisson
        self.merge_similarity = merge_similarity
        self.allow_proactive_sensitivity = allow_proactive_sensitivity

        self.num_alternative_states = num_alternative_states
        self.conf_sensitivity_drift = conf_sensitivity_drift
        self.conf_sensitivity_sustain = conf_sensitivity_sustain
        self.min_proactive_stdev = min_proactive_stdev
        self.alt_test_length = alt_test_length
        self.alt_test_period = alt_test_period


        self.sensitivity = sensitivity
        self.detector = make_detector(s = sensitivity)
        self.warn_detector = make_detector(s = self.get_warning_sensitivity(sensitivity))

        self.in_warning = False
        self.last_warning_point = 0
        self.waiting_for_concept_data = False

        self.max_state_id = 0

        self.random_state = None
        self._random_state = check_random_state(self.random_state)

        self.ex = -1
        self.classes = None
        self._train_weight_seen_by_model = 0
        self.similarity_measure = similarity_measure
        
        self.restore_state = None
        self.restore_state_set_point = 0
        self.restore_point_type = None
        self.allow_backtrack = allow_backtrack

        # Exposed 
        self.found_change = False
        self.num_states = 1
        self.active_state = self.max_state_id
        self.alternative_states = []
        self.reset_alternative_states = False
        self.states = []
        self.set_restore_state = None
        self.load_restore_state = None

        self.alternative_states_difference_confidence = {}
        self.signal_difference_backtrack = False
        self.signal_confidence_backtrack = False

        self.last_label = 0

        self.state_repository = {}
        self.testing_state_repository = {}
        self.testing_state_stats = {}
        init_id = self.max_state_id
        self.max_state_id += 1
        init_state = state(init_id, self.learner())
        self.state_repository[init_id] = init_state
        self.active_state_id = init_id

        self.recent_accuracy = []
        self.recent_non_masked_history = deque()
        self.history = []
        self.testing_state_history = []

        self.inactive_test_min = 15
        self.inactive_test_grace = 200


    def get_warning_sensitivity(self, s):
        return s * 10

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
        # if masked:
        #     return
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
        logging.debug(f"Partial fit on X: {X}, y:{y}, masked: {masked}, using state {self.active_state_id}")

        temporal_X = self.get_temporal_x(X)
        prediction = self.predict(temporal_X)[0]

        label = y if not masked else self.get_imputed_label(X=X, prediction = prediction, last_label = self.last_label)
        self.last_label = label

        # correctly_classified from the systems point of view.
        correctly_classifies = prediction == label

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

        backtrack_target = None
        found_change = False
        current_sensitivity = self.get_current_sensitivity()
        if not masked:
            self.recent_non_masked_history.append(self.history[-1])
            if len(self.recent_non_masked_history) > 500:
                self.recent_non_masked_history.popleft()
            if len(self.recent_non_masked_history) > self.inactive_test_min - 10:
                self.recent_accuracy.append(sum([1 if e['p'] == e['label'] else 0 for e in self.recent_non_masked_history]) / len(self.recent_non_masked_history))

            self.get_active_state().classifier.partial_fit(
                np.asarray([temporal_X]),
                np.asarray([label]),
                sample_weight=np.asarray([sample_weight])
            )

            if self.allow_backtrack:
                # Testing is expensive, so we only test for a length of time
                # after each restore point is placed.
                if (self.ex - self.restore_state_set_point) < self.alt_test_length:
                    backtrack_target = self.test_inactive_states(temporal_X, label, sample_weight, prediction)
            else:
                backtrack_target = None


            self.detector.delta = current_sensitivity
            self.warn_detector.delta = self.get_warning_sensitivity(current_sensitivity)

            self.detector.add_element(int(correctly_classifies))
            self.warn_detector.add_element(int(correctly_classifies))

            if self.warn_detector.detected_change():
                self.in_warning = True
                self.last_warning_point = self.ex
                self.warn_detector = make_detector(s = self.get_warning_sensitivity(current_sensitivity))

            found_change = self.detector.detected_change() or self.waiting_for_concept_data
        
        if found_change:
            ranked_alternatives, use_shadow, shadow_model, can_find_good_model = self.rank_inactive_models_for_suitability()
            if can_find_good_model:
                if use_shadow:
                    shadow_id, shadow_state = self.make_state()
                    shadow_state.classifier = shadow_model
                    self.state_repository[shadow_id] = shadow_state
                    self.active_state_id = shadow_id
                else:
                    transition_target_id = ranked_alternatives[-1]
                    self.active_state_id = transition_target_id
                self.waiting_for_concept_data = False

                self.detector = make_detector(s = current_sensitivity)
                self.warn_detector = make_detector(s = self.get_warning_sensitivity(current_sensitivity))
                self.recent_non_masked_history = deque()
                self.recent_accuracy = []


                self.place_restorepoint(ranked_alternatives, shadow_model, 'transition')
            else:
                self.waiting_for_concept_data = True
        elif backtrack_target is not None:
            restore_state = self.restore_state
            logging.debug(f"Restoring {restore_state.id}")
            self.state_repository[restore_state.id] = self.restore_state
            # print(backtrack_target)
            # print(self.testing_state_repository)
            if backtrack_target == -1:
                new_id, target_state = self.make_state()
                target_state.classifier = self.testing_state_repository[-1].classifier
                self.state_repository[new_id] = target_state
            else:
                # target_state = self.testing_state_repository[backtrack_target]
                target_state = self.state_repository[backtrack_target]
            self.active_state_id = target_state.id
            self.detector = make_detector(s = current_sensitivity)
            self.warn_detector = make_detector(s = self.get_warning_sensitivity(current_sensitivity))
            self.recent_non_masked_history = deque()
            self.recent_accuracy

            ranked_alternatives, use_shadow, shadow_model, can_find_good_model = self.rank_inactive_models_for_suitability()
            if can_find_good_model:
                self.place_restorepoint(ranked_alternatives, shadow_model, 'transition')
        else:
            should_place_restore_point = self.should_place_restore_point()
            if should_place_restore_point:
                ranked_alternatives, use_shadow, shadow_model, can_find_good_model = self.rank_inactive_models_for_suitability()
                if can_find_good_model:
                    self.place_restorepoint(ranked_alternatives, shadow_model, 'transition')



        # Set exposed info
        self.active_state = self.active_state_id
        self.found_change = found_change
        self.states = self.state_repository
        self.current_sensitivity = current_sensitivity




    
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
            state_predictions = state_predictions[-len(recent_window):]
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


        return [[x[0] for x in state_performace_by_id], use_shadow, shadow_model, True]


    def place_restorepoint(self, model_ids_to_test, shadow_model, restore_point_type):
        # Take a copy of the current state
        logging.debug("Placing Restore point")
        self.restore_state = deepcopy(self.get_active_state())
        self.restore_state_set_point = self.ex
        self.restore_point_type = restore_point_type

        self.testing_state_history = []
        self.testing_state_repository = {}
        self.testing_state_stats = {}

        if len(model_ids_to_test) < 0:
            for state_id in model_ids_to_test[-self.num_alternative_states]:
                if state_id == self.active_state_id:
                    continue
                self.testing_state_repository[state_id] = deepcopy(self.state_repository[state_id])
                self.testing_state_stats[state_id] = {'sustained_confidence_sum': 0, 'quick_change_detector': make_detector(s = self.conf_sensitivity_drift), "found_quick_change": False}
        
        self.testing_state_repository[-1] = state(-1, shadow_model)
        self.testing_state_stats[-1] = {'sustained_confidence_sum': 0, 'quick_change_detector': make_detector(s = self.conf_sensitivity_drift), "found_quick_change": False}
        logging.debug(self.testing_state_repository)
        logging.debug(self.testing_state_stats)


    def should_place_restore_point(self):
        """ Testing strategies could involve looking for local maxima,
        i.e. restoring to a known good point. For not just periodic.
        """
        # Check time since last restore point
        above_wait_period = (self.ex - self.restore_state_set_point) > self.alt_test_period

        return above_wait_period

    def test_inactive_states(self, temporal_X, label, sample_weight, p):
        backtrack_states = []
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

            test_history = [e for e in self.testing_state_history if e['active_model'] == inactive_state_id and not e['masked']]
            logging.debug(f"Seen {len(test_history)} samples")
            if len(test_history) < self.inactive_test_min:
                continue

            state_accuracy = sum([1 if e['p'] == e['label'] else 0 for e in test_history]) / len(test_history)
            main_state_accuracy = sum([1 if e['main_state_p'] == e['label'] else 0 for e in test_history]) / len(test_history)

            kappa_measure = (state_accuracy - main_state_accuracy) / (1 - main_state_accuracy) if main_state_accuracy < 1 else 0

            logging.debug(self.recent_accuracy[-100:])
            logging.debug(np.std(self.recent_accuracy[-100:]))
            # main_state_recent_acc_std = max(np.std(self.recent_accuracy[-100:]), 0.001)
            main_state_recent_acc_std = np.std(self.recent_accuracy[-100:])
            scaled_main_state_recent_acc_std = np.std(self.recent_accuracy[-100:]) / (1-main_state_accuracy) if main_state_accuracy < 1 else 0.0001

            logging.debug(f"testing state acc: {state_accuracy}")
            logging.debug(f"main state acc: {main_state_accuracy}")
            logging.debug(f"main state std: {main_state_recent_acc_std}")


            # probability_drawn_from_main_state = (1 - scipy.stats.norm(loc = main_state_accuracy, scale = main_state_recent_acc_std).cdf(state_accuracy))
            # probability_drawn_from_main_state = (1 - scipy.stats.norm(loc = main_state_accuracy, scale = scaled_main_state_recent_acc_std).cdf(state_accuracy))
            probability_drawn_from_main_state = (1 - scipy.stats.norm(loc = 0, scale = scaled_main_state_recent_acc_std ).cdf(kappa_measure))
            logging.debug(f"probability_drawn_from_main_state: {probability_drawn_from_main_state}")
            self.testing_state_stats[inactive_state_id]["sustained_confidence_sum"] += probability_drawn_from_main_state
            logging.debug(f"sustained_confidence_sum: {self.testing_state_stats[inactive_state_id]['sustained_confidence_sum']}")
            avg_sustained_confidence = self.testing_state_stats[inactive_state_id]["sustained_confidence_sum"] / len(test_history)
            logging.debug(f"avg_sustained_confidence: {avg_sustained_confidence}")

            self.testing_state_stats[inactive_state_id]["quick_change_detector"].add_element(main_state_accuracy - state_accuracy)

            self.testing_state_stats[inactive_state_id]["found_quick_change"] = (self.testing_state_stats[inactive_state_id]["quick_change_detector"].detected_change() and state_accuracy > main_state_accuracy) or self.testing_state_stats[inactive_state_id]["found_quick_change"]
            logging.debug(f"Quick change detected: {self.testing_state_stats[inactive_state_id]['found_quick_change']}")
            
            if len(test_history) > self.inactive_test_grace:
                if avg_sustained_confidence < self.conf_sensitivity_sustain:
                    logging.debug("Conf signal")
                    backtrack_states.append((inactive_state_id, state_accuracy))
                if self.testing_state_stats[inactive_state_id]["found_quick_change"] and state_accuracy > main_state_accuracy:
                    logging.debug("Drift signal")
                    backtrack_states.append((inactive_state_id, state_accuracy))
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
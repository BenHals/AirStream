import time
import logging
import pathlib
from copy import deepcopy

from ds_classifier import DSClassifier
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
import numpy as np

from skika.evaluation.inspect_recurrence import InspectPrequential

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

class ReverseTreeGenerator:
    def __init__(self, difficulty, reverse = False, tree_random_state = None, sample_random_state = None):
        self.stream = RandomTreeGenerator(max_tree_depth=difficulty+1, min_leaf_depth=difficulty, n_classes=2, n_cat_features=1, n_num_features=2, tree_random_state=tree_random_state, sample_random_state=sample_random_state)
        self.reverse = reverse
        self.tree_random_state = tree_random_state
        self.sample_random_state = sample_random_state
        self.n_samples = self.stream.n_samples
        self.n_targets = self.stream.n_targets
        self.n_num_features = self.stream.n_num_features + 1
        self.n_cat_features = self.stream.n_cat_features
        self.n_features = self.stream.n_features + 1
        self.n_classes = self.stream.n_classes
        self.cat_features_idx = self.stream.cat_features_idx
        self.feature_names = self.stream.feature_names + ['extra']
        self.target_names = self.stream.target_names
        self.target_values = self.stream.target_values
        self.name = self.stream.name
    
    def prepare_for_use(self):
        self.stream.prepare_for_use()
    
    def next_sample(self):
        X, y = self.stream.next_sample()
        extra_val = np.random.rand()
        X = np.append(X, [extra_val])
        if self.reverse:
            if extra_val > 0.5:
                y = [not y[0]]
        else:
            if extra_val < 0.5:
                y = [not y[0]]
        return X, y

    
    def n_remaining_samples(self):
        return self.stream.n_remaining_samples()
    
    def has_more_samples(self):
        return self.stream.has_more_samples()


def process(options):
    output_path = pathlib.Path.cwd() / 'synthetic_tests' / options['generator'] / f"{options['gen_seed']}${options['seed']}"
    name = '-'.join([f"{k:3}${v}" for k, v in options.items()])
    classifier = DSClassifier(
        learner=lambda : deepcopy(HoeffdingTree()),
        allow_backtrack=True,
        window = options["window"],
        sensitivity = options["sensitivity"],
        poisson = options["poisson"],
        num_alternative_states = options["num_alternative_states"],
        conf_sensitivity_drift = options["conf_sensitivity_drift"],
        conf_sensitivity_sustain = options["conf_sensitivity_sustain"],
        alt_test_length = options["alt_test_length"],
        alt_test_period = options["alt_test_period"]
        )
    output_path.mkdir(exist_ok=True, parents=True)
    if options['generator'] == 'reverse_tree':
        stream_A = ReverseTreeGenerator(options['difficulty'], False, options['gen_seed'], options["seed"])
        stream_B = ReverseTreeGenerator(options['difficulty'], True, options['gen_seed'], options["seed"])
    stream_A.prepare_for_use()
    stream_B.prepare_for_use()

    stream_AB = ConceptDriftStream(stream=stream_A, drift_stream=stream_B, position=options['positions'][0], width=options['widths'][0])
    stream_ABC = ConceptDriftStream(stream=stream_AB, drift_stream=stream_A, position=options['positions'][1], width=options['widths'][1])
    stream_ABC.prepare_for_use()

    save_X, save_y = stream_ABC.next_sample(options['run_length'])
    df = pd.DataFrame(np.hstack((save_X, np.array([save_y]).T)))
    df.to_csv(output_path / f"{name}_save.csv")

    stream = DataStream(df)
    stream.prepare_for_use()

    evaluator = InspectPrequential(show_plot=True, max_samples=run_length, data_expose={'drift_points': {options['positions'][0], options['positions'][1]}}, name = f'bt-{name}.pdf')
    evaluator.evaluate(stream=stream, model=classifier, model_names=["test"])


logging.basicConfig(level=logging.DEBUG, filename=f"logs/experiment-{time.time()}.log", filemode='w')
# run_length = 15000
# version = 1
# classifier = DSClassifier(learner=lambda : deepcopy(HoeffdingTree()), allow_backtrack=True, alt_test_length=1000, alt_test_period=1000)

# difficulty = 0
# trs = 30
# srs = 52
# stream_A = ReverseTreeGenerator(difficulty, False, trs, srs)
# # stream_A = STAGGERGenerator(classification_function=0)
# stream_A.prepare_for_use()
# stream_B = ReverseTreeGenerator(difficulty, True, trs, srs)
# # stream_B = STAGGERGenerator(classification_function=1)
# stream_B.prepare_for_use()

# stream_AB = ConceptDriftStream(stream=stream_A, drift_stream=stream_B, position=2000, width=1)
# stream_ABC = ConceptDriftStream(stream=stream_AB, drift_stream=stream_A, position=8000, width=5000)
# stream_ABC.prepare_for_use()

# evaluator = InspectPrequential(show_plot=True, max_samples=run_length, data_expose={'drift_points': {2000, 8000}}, name = f'bt-{version}.pdf')
# evaluator.evaluate(stream=stream_ABC, model=classifier, model_names=["test"])

options = {
    'generator': 'reverse_tree',
    'difficulty': 1,
    'gen_seed': 15
    'seed': 15,
    'positions': [2000, 8000],
    'widths': [1, 5000],
    'run_length': 15000,
    "window": 25,
    "sensitivity": 0.05,
    "poisson": 10,
    "num_alternative_states": 5,
    "conf_sensitivity_drift": 0.05,
    "conf_sensitivity_sustain": 0.125,
    "alt_test_length": 2000,
    "alt_test_period": 2000,
    'commit': get_git_revision_short_hash()
}

process(options)
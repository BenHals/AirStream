import time
import logging
import pathlib
from copy import deepcopy
import subprocess
import json
import pickle
import tqdm
import random
import multiprocessing as mp
import argparse
from math import ceil, floor
import time
import os
import psutil
import sys

from ds_classifier import DSClassifier
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.data_stream import DataStream
from simple_baseline import SimpleBaseline
from simple_window_baseline import SimpleWindowBaseline
from simple_baseline_skmf import SimpleBaselineSKMF
import numpy as np
import pandas as pd

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

class ReverseTreeGenerator:
    def __init__(self, difficulty, reverse = False, tree_random_state = None, sample_random_state = None):
        self.stream = RandomTreeGenerator(max_tree_depth=difficulty+1, min_leaf_depth=difficulty, n_classes=2, n_cat_features=3, n_num_features=4, tree_random_state=tree_random_state, sample_random_state=sample_random_state)
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
            if extra_val > 0.7:
                y = [not y[0]]
        # else:
        #     if extra_val < 0.5:
        #         y = [not y[0]]
        return X, y

    
    def n_remaining_samples(self):
        return self.stream.n_remaining_samples()
    
    def has_more_samples(self):
        return self.stream.has_more_samples()
class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)
class ReverseRBFGenerator:
    def __init__(self, difficulty, reverse = False, tree_random_state = None, sample_random_state = None):
        self.stream = RandomRBFGenerator(n_centroids=difficulty*6, n_classes=4, n_features=8, model_random_state=tree_random_state, sample_random_state=sample_random_state)
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
            if extra_val > 0.3:
                y = [not y[0]]
        # else:
        #     if extra_val < 0.5:
        #         y = [not y[0]]
        return X, y

    
    def n_remaining_samples(self):
        return self.stream.n_remaining_samples()
    
    def has_more_samples(self):
        return self.stream.has_more_samples()


def get_concept_transparency_data(ground_truth, system, purity, merge_key):
    print(ground_truth[:5])
    print(system[:5])
    print(purity[:5])

    gt_values, gt_total_counts = np.unique(ground_truth, return_counts = True)
    sys_values, sys_total_counts = np.unique(system, return_counts = True)

    print(gt_values)
    print(sys_values)
    matrix = np.array([ground_truth, system, purity]).transpose()
    print(matrix[:5])
    # Key = (gt_concept, sys_concept)
    recall_values = {}
    precision_values = {}
    gt_results = {}
    sys_results = {}
    overall_results = {
        'Max Recall': 0,
        'Max Precision': 0,
        'Precision for Max Recall': 0,
        'Recall for Max Precision': 0,
        'f1' : 0,
        'MR by System': 0,
        'MP by System': 0,
        'PMR by System': 0,
        'RMP by System': 0,
        'f1 by System': 0,
        'Num Good System Concepts': 0,
    }
    gt_proportions = {}
    sys_proportions = {}
    
    for gt_i, gt in enumerate(gt_values):
        gt_total_count = gt_total_counts[gt_i]
        gt_mask = matrix[matrix[:,0] == gt]
        sys_by_gt_values, sys_by_gt_counts = np.unique(gt_mask[:, 1], return_counts = True)
        print(f"GT SHOULD BE THE SAME {gt_total_count}:{gt_mask.shape[0]}")
        gt_proportions[gt] = gt_mask.shape[0] / matrix.shape[0]
        max_recall = None
        max_recall_sys = None
        max_precision = None
        max_precision_sys = None
        max_f1 = None
        max_f1_sys = None
        for sys_i,sys in enumerate(sys_by_gt_values):
            sys_by_gt_count = sys_by_gt_counts[sys_i]
            sys_total_count = sys_total_counts[sys_values.tolist().index(sys)]
            # recall_rows = gt_mask[(gt_mask[:,1] == sys)]
            # print(f"SYS SHOULD BE THE SAME {sys_by_gt_count}:{recall_rows.shape[0]}")

            if gt_total_count != 0:
                recall = sys_by_gt_count / gt_total_count
            else:
                recall = 1

            recall_values[(gt, sys)] = recall

            # sys_mask = matrix[matrix[:,1] == sys]
            # sys_proportions[sys] = sys_mask.shape[0] / matrix.shape[0]
            sys_proportions[sys] = sys_total_count / matrix.shape[0]
            # precision_rows = sys_mask[sys_mask[:,0] == gt]
            # if sys_mask.shape[0] != 0:
            #     precision = precision_rows.shape[0] / sys_mask.shape[0]
            # else:
            #     precision = 1
            if sys_total_count != 0:
                precision = sys_by_gt_count / sys_total_count
            else:
                precision = 1
            precision_values[(gt, sys)] = precision

            f1 = 2 * ((recall * precision) / (recall + precision))

            if max_recall == None or recall > max_recall:
                max_recall = recall
                max_recall_sys = sys
            if max_precision == None or precision > max_precision:
                max_precision = precision
                max_precision_sys = sys
            if max_f1 == None or f1 > max_f1:
                max_f1 = f1
                max_f1_sys = sys
        precision_max_recall = precision_values[(gt, max_recall_sys)]
        recall_max_precision = recall_values[(gt, max_precision_sys)]
        print(" GT recalls")
        # print(recall_values)
        print("GT precisions")
        # print(precision_values)
        gt_result = {
            'Max Recall': max_recall,
            'Max Precision': max_precision,
            'Precision for Max Recall': precision_max_recall,
            'Recall for Max Precision': recall_max_precision,
            'f1': max_f1
        }
        print(" GT result")
        print(gt_result)
        gt_results[gt] = gt_result
        overall_results['Max Recall'] += max_recall
        overall_results['Max Precision'] += max_precision
        overall_results['Precision for Max Recall'] += precision_max_recall
        overall_results['Recall for Max Precision'] += recall_max_precision
        overall_results['f1'] += max_f1

    
    for sys in sys_values:
        max_recall = None
        max_recall_gt = None
        max_precision = None
        max_precision_gt = None
        max_f1 = None
        max_f1_sys = None
        for gt in gt_values:
            if (gt, sys) not in recall_values:
                continue
            if (gt, sys) not in precision_values:
                continue
            recall = recall_values[(gt, sys)]
            precision = precision_values[(gt, sys)]

            f1 = 2 * ((recall * precision) / (recall + precision))

            if max_recall == None or recall > max_recall:
                max_recall = recall
                max_recall_gt = gt
            if max_precision == None or precision > max_precision:
                max_precision = precision
                max_precision_gt = gt
            if max_f1 == None or f1 > max_f1:
                max_f1 = f1
                max_f1_sys = sys

        precision_max_recall = precision_values[(max_recall_gt, sys)]
        recall_max_precision = recall_values[(max_precision_gt, sys)]   
        sys_result = {
            'Max Recall': max_recall,
            'Max Precision': max_precision,
            'Precision for Max Recall': precision_max_recall,
            'Recall for Max Precision': recall_max_precision,
            'f1': max_f1
        }
        print("Sys result")
        print(sys_result)
        sys_results[sys] = sys_result     
        overall_results['MR by System'] += max_recall * sys_proportions[sys]
        overall_results['MP by System'] += max_precision * sys_proportions[sys]
        overall_results['PMR by System'] += precision_max_recall * sys_proportions[sys]
        overall_results['RMP by System'] += recall_max_precision * sys_proportions[sys]
        overall_results['f1 by System'] += max_f1 * sys_proportions[sys]
        if max_recall > 0.75 and precision_max_recall > 0.75:
            overall_results['Num Good System Concepts'] += 1

    overall_results['Max Recall'] /= gt_values.size
    overall_results['Max Precision'] /= gt_values.size
    overall_results['Precision for Max Recall'] /= gt_values.size
    overall_results['Recall for Max Precision'] /= gt_values.size
    overall_results['f1'] /= gt_values.size
    # overall_results['MR by System'] /= sys_values.size
    # overall_results['MP by System'] /= sys_values.size
    # overall_results['PMR by System'] /= sys_values.size
    # overall_results['RMP by System'] /= sys_values.size
    return gt_results, overall_results


def get_option_filename(options):
    string_repr = ""
    for k,v in options.items():
        if k not in options['experiment_varying_parameters']:
            continue
        value_str = str(v).replace(" ", "")
        for c in ['\\', '/', ':', '*', '"', '<', '>', '|']:
            value_str = value_str.replace(c, '&')
        string_repr += f"{k[:3]}${value_str}+"
    return string_repr

def process(options):

    output_path = options['run_directory'].parent
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
        alt_test_period = options["alt_test_period"],
        drift_detector = options["dd"] if 'dd' in options else 'adwin'
        )
    nobt_classifier = DSClassifier(
        learner=lambda : deepcopy(HoeffdingTree()),
        allow_backtrack=False,
        window = options["window"],
        sensitivity = options["sensitivity"],
        poisson = options["poisson"],
        num_alternative_states = options["num_alternative_states"],
        conf_sensitivity_drift = options["conf_sensitivity_drift"],
        conf_sensitivity_sustain = options["conf_sensitivity_sustain"],
        alt_test_length = options["alt_test_length"],
        alt_test_period = options["alt_test_period"],
        drift_detector = options["dd"] if 'dd' in options else 'adwin'
        )
    df = pd.read_csv(str(output_path / 'save_data.csv'), index_col=0)
    df_copy = pd.read_csv(str(output_path / 'save_data.csv'), index_col=0)
    length = df.shape[0]
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
    def get_results(stream, classifier, path, length, name, train_window = 0):
        predictions = []
        labels = []
        states = []
        amend_states = []
        gts = []
        right = 0
        wrong = 0
        start_time = time.time()
        mem = 0
        for ex in tqdm.tqdm(range(length)):
            X,y = stream.next_sample()
            if ex >= train_window:
                p = classifier.predict(X)
                predictions.append(int(p[0]))
                labels.append(int(y[0]))
                if p[0] == y[0]:
                    right += 1
                else:
                    wrong += 1
                
                try:
                    state = classifier.active_state_id
                except Exception as e:
                    state = 0
                states.append(state)
                amend_states.append(state)

                for seg in options['boundaries']:
                    if seg[1] <= ex < seg[2]:
                        gt = seg[0]
                        break
                gts.append(gt)

                if hasattr(classifier, 'restore_history_amend'):
                    if classifier.restore_history_amend is not None:
                        for ai in range(classifier.restore_history_amend['start'], classifier.restore_history_amend['end']):
                            if amend_states[ai] == classifier.restore_history_amend['model_from']:
                                amend_states[ai] = classifier.restore_history_amend['model_to']

            # process = psutil.Process(os.getpid())
            classifier.partial_fit(X, y)
            # mem = max(mem, process.memory_info().rss)
            # if hasattr(classifier, 'state_repository'):
            #     current_mem = get_size(classifier.state_repository)
            # elif hasattr(classifier, 'ensemble'):
            #     current_mem = get_size(classifier.ensemble)
            # elif hasattr(classifier, 'size'):
            #     # print(classifier.size)
            #     current_mem = classifier.size
            # elif hasattr(classifier, 'classifier'):
            #     # print("mem is classifier")
            #     current_mem = get_size(classifier.classifier)
            # else:
            #     current_mem = get_size(classifier)
            # print(current_mem)
            # mem = max(mem, current_mem)
            # print(mem)
        end_time = time.time()
        gt_results, overall_results = get_concept_transparency_data(np.array(gts), np.array(states), np.zeros(len(states)), {})
        gt_results_a, overall_results_a = get_concept_transparency_data(np.array(gts), np.array(amend_states), np.zeros(len(amend_states)), {})
        result = {
            'predictions': predictions,
            'labels': labels,
            'states': states,
            'gts': gts,
            'accuracy': right / (right + wrong),
            'gt_results': gt_results,
            'gt_results_a': gt_results_a,
            'overall_results': overall_results,
            'overall_results_a': overall_results_a,
            'time': end_time - start_time,
        }
        # if hasattr(classifier, 'state_repository'):
        #     result['mem'] = get_size(classifier.state_repository)
        # elif hasattr(classifier, 'ensemble'):
        #     result['mem'] = get_size(classifier.ensemble)
        # elif hasattr(classifier, 'classifier'):
        #     print("mem is classifier")
        #     result['mem'] = get_size(classifier.classifier)
        # else:
        #     result['mem'] = get_size(classifier)
        # result['mem'] = mem
        if hasattr(classifier, 'state_repository'):
            current_mem = get_size(classifier.state_repository)
        elif hasattr(classifier, 'ensemble'):
            current_mem = get_size(classifier.ensemble)
        elif hasattr(classifier, 'size'):
            # print(classifier.size)
            current_mem = classifier.size
        elif hasattr(classifier, 'classifier'):
            # print("mem is classifier")
            current_mem = get_size(classifier.classifier)
        else:
            current_mem = get_size(classifier)
        result['mem'] = current_mem
        print(current_mem)

            
        # try:
        #     result['max_mem'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # except:
        #     result['max_mem'] = mem

        result_path = path / f"{name}_results.txt"
        with result_path.open('w') as f:
            for k in result:
                f.write(f"{k}:{str(result[k])}\n")
        
        result_path = path / f"{name}_results.pickle"
        with result_path.open('wb') as f:
            pickle.dump(result, f)

    # stream = DataStream(df)
    # stream.prepare_for_use()
    # get_results(stream, classifier, output_path, length, 'AirStreamBacktrackAmend_AM')

    # stream = DataStream(df)
    # stream.prepare_for_use()
    # get_results(stream, nobt_classifier, output_path, length, 'AirStreamNoBacktrack_AM')

    if not options['run_baselines']:
        return
    x_locs = None
    y_loc = None
    spacial_pattern = [(random.random(), random.random(), random.random(), random.random()) for i in range(df.shape[1])]
    bins = [0, 0.5]
    # linear_classifier = SimpleBaseline('linear', x_locs, y_loc, spacial_pattern, bins)
    # stream = DataStream(df)
    # stream.prepare_for_use()
    # get_results(stream, linear_classifier, output_path, length, 'IDW_AM')

    # norm_classifier = SimpleBaseline('normSCG', x_locs, y_loc, spacial_pattern, bins)
    # stream = DataStream(df)
    # stream.prepare_for_use()
    # get_results(stream, norm_classifier, output_path, length, 'Gaussian_AM')

    # nc_classifier = SimpleBaseline('temporal', x_locs, y_loc, spacial_pattern, bins)
    # stream = DataStream(df)
    # stream.prepare_for_use()
    # get_results(stream, nc_classifier, output_path, length, 'NC_AM')

    # OK_classifier = SimpleBaseline('OK', x_locs, y_loc, spacial_pattern, bins)
    # stream = DataStream(df)
    # stream.prepare_for_use()
    # get_results(stream, OK_classifier, output_path, length, 'OK_AM')

    # tree_classifier = SimpleWindowBaseline('tree', x_locs, y_loc, spacial_pattern)
    # stream = DataStream(df)
    # stream.prepare_for_use()
    # get_results(stream, tree_classifier, output_path, length, 'RF_AM', 4000)
    # tree_classifier = SimpleWindowBaseline('tree', x_locs, y_loc, spacial_pattern)
    # stream = DataStream(df)
    # stream.prepare_for_use()
    # get_results(stream, tree_classifier, output_path, length, 'RF2_AM', 2000)
    tree_classifier = SimpleWindowBaseline('tree', x_locs, y_loc, spacial_pattern)
    stream = DataStream(df)
    stream.prepare_for_use()
    # df_pretrain= df_copy.sample(2)
    # stream_pretrain = DataStream(df_pretrain)
    # stream_pretrain.prepare_for_use()
    if options['generator'] == 'reverse_tree':
        stream_A = ReverseTreeGenerator(options['difficulty'], False, options['gen_seed'], options["seed"])
        stream_B = ReverseTreeGenerator(options['difficulty'], True, options['gen_seed'], options["seed"])
    if options['generator'] == 'reverse_RBF':
        stream_A = ReverseRBFGenerator(options['difficulty'], False, options['gen_seed'], options["seed"])
        stream_B = ReverseRBFGenerator(options['difficulty'], True, options['gen_seed'], options["seed"])
    stream_A.prepare_for_use()
    stream_B.prepare_for_use()
    for i in range(100000):
        X, y = stream_A.next_sample()
        X, y = stream_B.next_sample()
    for i in range(100):
        X, y = stream_A.next_sample()
        tree_classifier.partial_fit([X], [y])
    for i in range(100):
        X, y = stream_B.next_sample()
        tree_classifier.partial_fit([X], [y])
    get_results(stream, tree_classifier, output_path, length, 'RFPT_AM_Both', 0)

    # arf_classifier = SimpleBaselineSKMF('arf', x_locs, y_loc, spacial_pattern, bins)
    # stream = DataStream(df)
    # stream.prepare_for_use()
    # get_results(stream, arf_classifier, output_path, length, 'ARF_AM')


def get_state_boundaries(options):
    max_state_id = max(options['gts']) + 1
    partial_state_map = {}
    boundaries = []
    active_state = [options['gts'][0], 0, None]
    for di,drift in enumerate(options['positions']):
        width = options['widths'][di]
        next_state = options['gts'][di + 1]
        if width == 1:
            active_state[2] = drift
            boundaries.append(active_state)
            active_state = [next_state, drift, None]
        else:
            partial_A_start = drift - ceil(width / 2)
            partial_A_end = drift + floor(width / 2)
            partial_B_start = drift
            partial_B_end = drift + floor(width / 2)
            active_state[2] = partial_A_start
            boundaries.append(active_state)

            partial_A = f"{active_state[0]}-{next_state}-A"
            if partial_A in partial_state_map:
                partial_A_id = partial_state_map[partial_A]
            else:
                partial_A_id = max_state_id
                max_state_id += 1
                partial_state_map[partial_A] = partial_A_id
            # partial_B = f"{active_state[0]}-{next_state}-B"
            # if partial_B in partial_state_map:
            #     partial_B_id = partial_state_map[partial_B]
            # else:
            #     partial_B_id = max_state_id
            #     max_state_id += 1
            #     partial_state_map[partial_B] = partial_B_id
            
            boundaries.append([partial_A_id, partial_A_start, partial_A_end])
            # boundaries.append([partial_B_id, partial_B_start, partial_B_end])
            active_state = [next_state, partial_A_end, None]
    active_state[2] = options['run_length']
    boundaries.append(active_state)

    return boundaries


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--cpu', default=2, type=int)
    my_parser.add_argument('--single', action='store_true')
    my_parser.add_argument('--directory', default="cwd")

    args = my_parser.parse_args()
    logging.basicConfig(level=logging.INFO, filename=f"logs/experiment-{time.time()}.log", filemode='w')

    option_files = pathlib.Path(args.directory).rglob('options.json')
    options = []
    for of in option_files:
        o = json.load(of.open())
        o['run_directory'] = of
        o['run_baselines'] = True
        options.append(o)

    if args.single:
        process(options[0])
    else:
        pool = mp.Pool(processes=args.cpu)
        results = pool.map(process, options, chunksize=1)
from copy import deepcopy
import pathlib
import argparse
import logging
import time
import os
import io
import json
from itertools import product
import pickle
import subprocess
import time
import copy

from collections import namedtuple
from collections import deque
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.data.data_stream import DataStream
from scipy.io import arff

from simple_baseline import SimpleBaseline
from simple_window_baseline import SimpleWindowBaseline
from simple_baseline_skmf import SimpleBaselineSKMF
from csv_result_extraction import log_accuracy_from_df


import sklearn
import sklearn.ensemble
import sklearn.svm
import sklearn.naive_bayes
import sklearn.model_selection
import sklearn.metrics
import sklearn.dummy

from create_dataset import create_dataset

from ds_classifier import DSClassifier

# RunOptions = namedtuple('RunOptions', 'directory header filename backtrack proactive_sensitivity cl an window', defaults=[50])
RunOptions = namedtuple('RunOptions', 'directory header filename ct seed')

logging.basicConfig(level=logging.INFO, filename=f"logs/experiment-{time.time()}.log", filemode='w')

def get_git_revision_short_hash():
    try:
        h = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        h = "0"
    return h

class ModelExposure:
    """ Extracts information made availiable by a classifier
        on the current internat state.
    """
    def __init__(self, model):
        self.model = model

    def get_exposed_info_at_sample(self, sample_index):
        sample_exposed_info = {}
        if hasattr(self.model, 'found_change'):
            sample_exposed_info['found_change'] = self.model.found_change
        else:
            sample_exposed_info['found_change'] = -1
        if hasattr(self.model, 'active_state'):
            sample_exposed_info['active_state'] = self.model.active_state
        else:
            sample_exposed_info['active_state'] = -1
        if hasattr(self.model, 'reset_alternative_states'):
            sample_exposed_info['reset_alternative_states'] = self.model.reset_alternative_states
        else:
            sample_exposed_info['reset_alternative_states'] = -1
        if hasattr(self.model, 'alternative_states'):
            sample_exposed_info['alternative_states'] = self.model.alternative_states
        else:
            sample_exposed_info['alternative_states'] = -1
        if hasattr(self.model, 'states'):
            sample_exposed_info['states'] = self.model.states
        else:
            sample_exposed_info['states'] = -1
        if hasattr(self.model, 'set_restore_state'):
            sample_exposed_info['set_restore_state'] = self.model.set_restore_state
        else:
            sample_exposed_info['set_restore_state'] = -1
        if hasattr(self.model, 'load_restore_state'):
            sample_exposed_info['load_restore_state'] = self.model.load_restore_state
        else:
            sample_exposed_info['load_restore_state'] = [-1, -1]
        if hasattr(self.model, 'alternative_states_difference_confidence'):
            sample_exposed_info['alternative_states_difference_confidence'] = self.model.alternative_states_difference_confidence
        else:
            sample_exposed_info['alternative_states_difference_confidence'] = -1
        if hasattr(self.model, 'signal_confidence_backtrack'):
            sample_exposed_info['signal_confidence_backtrack'] = self.model.signal_confidence_backtrack
        else:
            sample_exposed_info['signal_confidence_backtrack'] = -1
        if hasattr(self.model, 'signal_difference_backtrack'):
            sample_exposed_info['signal_difference_backtrack'] = self.model.signal_difference_backtrack
        else:
            sample_exposed_info['signal_difference_backtrack'] = -1
        if hasattr(self.model, 'current_sensitivity'):
            sample_exposed_info['current_sensitivity'] = self.model.current_sensitivity
        else:
            sample_exposed_info['current_sensitivity'] = -1

        return sample_exposed_info


class RunStatistics:
    def __init__(self):
        self.right = 0
        self.wrong = 0
        self.sliding_window = deque()
        self.sliding_window_length = 300
        self.sliding_window_accuracy = 0
        self.sliding_window_sum = 0
    
    def add_observation(self, y, p):
        y = y[0]
        correct = y == p
        if correct:
            self.right += 1
            self.sliding_window.append(1)
            self.sliding_window_sum += 1
        else:
            self.wrong += 1
            self.sliding_window.append(0)
        
        if len(self.sliding_window) > self.sliding_window_length:
            self.sliding_window_sum -= self.sliding_window.popleft()
        
        self.sliding_window_accuracy = self.sliding_window_sum / len(self.sliding_window)

        return correct

    def get_seen(self):
        return self.wrong + self.right
    def get_accuracy(self):
        if self.get_seen() == 0:
            return 0
        return self.right / (self.get_seen())
    
def create_classifier(run_options, supplementary_info, sp_info):
    if run_options.ct == 'linear':
        sensor_locs = [None, None] if 'sensors' not in supplementary_info else supplementary_info['sensors']
        bins = None if 'bins' not in supplementary_info else supplementary_info['bins']
        classifier = SimpleBaseline(classifer_type = 'linear', x_locs = sensor_locs[0], y_loc = sensor_locs[1], spacial_pattern = sp_info, bins=bins)
    elif run_options.ct == 'normSCG':
        sensor_locs = [None, None] if 'sensors' not in supplementary_info else supplementary_info['sensors']
        bins = None if 'bins' not in supplementary_info else supplementary_info['bins']
        classifier = SimpleBaseline(classifer_type = 'normSCG', x_locs = sensor_locs[0], y_loc = sensor_locs[1], spacial_pattern = sp_info, bins=bins)
    elif run_options.ct == 'temporal':
        bins = None if 'bins' not in supplementary_info else supplementary_info['bins']
        classifier = SimpleBaseline(classifer_type = 'temporal', x_locs = [], y_loc = None, spacial_pattern = sp_info, bins=bins)
    elif run_options.ct == 'OK':
        bins = None if 'bins' not in supplementary_info else supplementary_info['bins']
        classifier = SimpleBaseline(classifer_type = 'OK', x_locs = [], y_loc = None, spacial_pattern = sp_info, bins=bins)
    elif run_options.ct == 'tree':
        classifier = SimpleWindowBaseline(classifer_type = 'tree', x_locs = [], y_loc = None, spacial_pattern = sp_info)
    elif run_options.ct == 'arf':
        bins = None if 'bins' not in supplementary_info else supplementary_info['bins']
        classifier = SimpleBaselineSKMF(classifer_type = 'arf', x_locs = [], y_loc = None, spacial_pattern = sp_info, bins = bins)
    else:
        classifier = DSClassifier(
                                    learner=HoeffdingTree,
                                    window=run_options.window,
                                    alt_test_length = run_options.atl * run_options.atp,
                                    alt_test_period = run_options.atp,
                                    sensitivity = run_options.bs,
                                    allow_backtrack=run_options.backtrack,
                                    conf_sensitivity_drift = run_options.csd,
                                    conf_sensitivity_sustain = run_options.css)
    return classifier

def write_to_file(results_filename, run_options, rows):
    with open(results_filename, 'a') as f:
        for r in rows:
            print(','.join([str(x) for x in r]), file = f)

def clear_results_file(results_filename):
    if pathlib.Path(results_filename).resolve().exists():
        os.remove(results_filename)

def run_stream_with_options(run_options, stream_file, parent_dir, load_arff = False, train_only_period = 0):
    logging.debug(f"Opeining file {stream_file} with parent {parent_dir} and options {run_options}")
    print(f"Opeining file {stream_file} with parent {parent_dir} and options {run_options}")
    results_filename = f"{parent_dir}{os.sep}{run_options.filename}.csv"
    results_filename_no_filetype = f"{parent_dir}{os.sep}{run_options.filename}"
    merges_filename = f"{parent_dir}{os.sep}{run_options.filename.split('.')[0]}-merges.pickle"
    info_filename = f"{parent_dir}{os.sep}{'.'.join(run_options.filename.split('.')[:-1]) if '.' in run_options.filename else run_options.filename}-run_stats.txt"
    

    # Check for existing run, if that run is over
    # a minimum size (i.e not started then cancelled)
    if os.path.exists(results_filename):
        print("Existing")
        print(os.path.getsize(results_filename))
        print(os.path.getsize(stream_file))
        if os.path.getsize(results_filename) > (os.path.getsize(stream_file) * 0.15):
            return results_filename, merges_filename, info_filename
    if os.path.exists(results_filename_no_filetype):
        print("Existing")
        print(os.path.getsize(results_filename_no_filetype))
        print(os.path.getsize(stream_file))
        if os.path.getsize(results_filename_no_filetype) > (os.path.getsize(stream_file) * 0.15):
            return results_filename, merges_filename, info_filename

    with open(f"{parent_dir}{os.sep}{run_options.filename}_run_options.txt", 'w+') as f:
        json.dump({**run_options._asdict(), "gitcommit": get_git_revision_short_hash()}, f)

    supplementary_filenames = list(parent_dir.glob('*_supp.txt'))
    supplementary_info = {}
    if len(supplementary_filenames) > 0:
        with open(supplementary_filenames[0], 'rb') as f:
            supplementary_info = json.loads(f.readline())
    supplementary_filenames = list(parent_dir.glob('df_info.json'))
    if len(supplementary_filenames) > 0:
        with open(supplementary_filenames[0], 'rb') as f:
            supplementary_info = {**json.loads(f.readline()), **supplementary_info}
    print(supplementary_info)
    sp_filenames = list(parent_dir.glob('spacial_pattern.json'))
    sp_info = None
    if len(sp_filenames) > 0:
        with open(sp_filenames[0], 'r') as f:
            sp_info = json.loads(f.readline())
    print(sp_info)

    #Run!
    start_time = time.time()
    print(results_filename)
    subprocess.run(["java", r'-javaagent:G:\My Drive\UniMine\Uni\PhD\AirStream\JournalRevision\AdditionalComparisons\dynse-master\dynse\target\sizeofag-1.0.0.jar', "-jar", r"G:\My Drive\UniMine\Uni\PhD\AirStream\JournalRevision\AdditionalComparisons\dynse-master\dynse\target\dynse-0.2-jar-with-dependencies.jar", "-t", "-f", str(stream_file), f"{run_options.filename}.csv"])
    end_time = time.time()
    time_taken = end_time - start_time


    with open(info_filename, 'w+') as f:
        print(f"time_taken: {time_taken}", file=f)

    return results_filename, merges_filename, info_filename

        

def gen_filename(run_options):
    return f"results.csv"

def get_stream_files(directory):
    base_directory_path = pathlib.Path(directory)
    print(base_directory_path)
    if not base_directory_path.exists():
        raise ValueError("Directory does not exist")
    
    csv_stream_files = []
    arf_stream_files = list(base_directory_path.glob('**/*.arff'))

    streams_by_parent_folder = {}
    for stream_file in [*csv_stream_files, *arf_stream_files]:
        parent_key = str(stream_file.parent.parent)
        if parent_key not in streams_by_parent_folder:
            streams_by_parent_folder[parent_key] = []
        streams_by_parent_folder[parent_key].append(stream_file)
    
    if len(streams_by_parent_folder) > 1:
        smallest_parent = min(*[len(streams_by_parent_folder[k]) for k in streams_by_parent_folder])
        stream_files = [sf for tup in zip(*[streams_by_parent_folder[k][:smallest_parent] for k in streams_by_parent_folder]) for sf in tup]
        for k in streams_by_parent_folder:
            stream_files += streams_by_parent_folder[k][smallest_parent:]
    else:
        stream_files = [*csv_stream_files, *arf_stream_files]
    print(stream_files)
    
    return stream_files
def get_concept_transparency_data(ground_truth, system, purity, merge_key):
    print(ground_truth[:5])
    print(system[:5])
    print(purity[:5])
    print(ground_truth.shape)
    print(system.shape)
    print(purity.shape)

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
def gridsearch_experiment(directory, header, results_filename, classifier_type, train_only_period, real_world_dataset_name, real_world_dataset_index, aux_feature_predictions, seed, using_synthetic_data=False, passed_aux=None,gts=None):
    results = {}
    stream_files = get_stream_files(directory)
    print(stream_files)
    aux_data = passed_aux if passed_aux is not None else aux_info
    for stream_file in stream_files:
        print(stream_file)
        base_fn = results_filename if results_filename is not None else gen_filename({})
        filename = f"{'.'.join(base_fn.split('.')[:-1]) if '.' in base_fn else base_fn}-{train_only_period}"
        run_options = RunOptions(str(directory), header, filename, classifier_type, seed)
        logging.basicConfig(level=logging.INFO, filename= str(stream_file.parent /f"{filename}-{time.time()}.log"), filemode='w')
        start_time = time.time()
        fn, mg, info = run_stream_with_options(run_options, stream_file, stream_file.parent, load_arff= '.ARFF' in str(stream_file), train_only_period = train_only_period)
        end_time = time.time()
        r = extract_results_from_csv(fn, mg, info, stream_file.parent, True, aux_data, results_timestamps, aux_feature_predictions)
        if using_synthetic_data:
            data = pd.read_csv(fn, header = 0)
            predictions = data['p'].values
            labels = data['y'].values
            gts=gts[-labels.shape[0]:]
            states = data['system_concept'].values
            gt_results, overall_results = get_concept_transparency_data(np.array(gts), np.array(states), np.zeros(len(states)), {})
            # gt_results_a, overall_results_a = get_concept_transparency_data(np.array(gts), np.array(amend_states), np.zeros(len(amend_states)), {})
            result = {
                'predictions': list(predictions),
                'labels': list(labels),
                'states': list(states),
                'gts': gts,
                'gt_results': gt_results,
                # 'gt_results_a': gt_results_a,
                'overall_results': overall_results,
                # 'overall_results_a': overall_results_a,
                'time': end_time - start_time,
                'mem': data['memory']
            }
            # result_path = results_dump_path / f"{name}_results.txt"
            result_path = stream_file.parent / f"{str(fn).replace('.csv', '')}_results.txt"
            with result_path.open('w') as f:
                for k in result:
                    f.write(f"{k}:{str(result[k])}\n")
            
            result_path = stream_file.parent / f"{str(fn).replace('.csv', '')}_results.pickle"
            with result_path.open('wb') as f:
                pickle.dump(result, f)
        results[json.dumps({**run_options._asdict(), "dataset_name": real_world_dataset_name, "dataset_target_index": real_world_dataset_index})] = r
        results_dump_path = stream_file.parent / f"{str(fn).replace('.csv', '')}-results.json"
        with results_dump_path.open('w') as f:
            json.dump(r, f)
    return results

def link(results, aux_info, results_timestamps, aux_feature_predictions, parent):
    timestamps = results_timestamps.iloc[-results.shape[0]:]
    print(timestamps)
    print(aux_info['date_time'])
    results.index = timestamps
    results.index = pd.to_datetime(timestamps, utc =False).dt.tz_localize(None)
    aux_info['date_time'] = pd.to_datetime(aux_info['date_time'], utc=False).dt.tz_localize(None)
    print(aux_info['date_time'])
    print(results.index)
    full_results = pd.merge(results, aux_info, how = 'left', left_index=True, right_on='date_time')
    full_res_fn = parent / "full_link.csv"
    full_results.to_csv(full_res_fn)
        
    accuracy = full_results.iloc[-1]['overall_accuracy']
    dummy = sklearn.dummy.DummyClassifier(strategy='most_frequent')
    dummy_accuracy = np.mean(sklearn.model_selection.cross_val_score(dummy, full_results['y'], full_results['y'], scoring='accuracy'))
    results = {'link-accuracy': {'dummy_score': dummy_accuracy, 'model_score': accuracy, 'dummy_increase': accuracy - dummy_accuracy, "feature_increase": -1}}

    test_cols = [c for c in aux_info.columns if c != "date_time"]
    for test_feature in test_cols:
        print(test_feature)
        aux_feature_prediction_score = aux_feature_predictions[test_feature] if test_feature in aux_feature_predictions else 0
        if 'system_concept' in full_results.columns:
            pred = full_results[[test_feature, 'system_concept']]
            y = pred[test_feature]
            if len(np.unique(y)) > 16:
                y = pd.qcut(pred[test_feature], 4, labels=False, duplicates="drop").values
            x = pred['system_concept'].values.reshape(-1, 1)
            print(x)

            dummy_scores = []
            model_scores = []

            for i in range(3):
                model = sklearn.ensemble.RandomForestClassifier(n_estimators = 10)
                dummy = sklearn.dummy.DummyClassifier(strategy = "prior")
                try:
                    dummy_score = sklearn.model_selection.cross_val_score(dummy, x, y, scoring='accuracy', cv = 5)
                    model_score = sklearn.model_selection.cross_val_score(model, x, y, scoring='accuracy', cv = 5)
                except Exception as e:
                    print(e)
                    continue
                dummy_scores.append(np.mean(dummy_score))
                model_scores.append(np.mean(model_score))
            dummy_score = np.mean(dummy_scores)
            model_score = np.mean(model_scores)
            dummy_increase = model_score - dummy_score
            print(f"Feature: {test_feature}, using state gave increase of {dummy_increase} in prediction accuracy")
            print(f"From {dummy_score} to {model_score} accuracy")
        else:
            dummy_score = 0
            model_score = 0
            dummy_increase = 0
        results[f"link-{test_feature}"] = {'dummy_score': dummy_score, 'model_score': model_score, 'dummy_increase': dummy_increase, 'feature_increase': model_score - aux_feature_prediction_score}
    return results
    
def extract_results_from_csv(file, merge_file, info_file, parent, readall, aux_info, results_timestamps, aux_feature_predictions):
    version = 1
    name_stem = pathlib.Path(file).stem
    # row_names = ['example', 'sliding_window_accuracy','is_correct','p','y','overall_accuracy','system_concept','change_detected','set_restore_state','_load_restore_state','signal_confidence_backtrack','signal_difference_backtrack','current_sensitivity', 'mask']
    row_names = ['example','sliding_window_accuracy','is_correct','p','y','overall_accuracy','evaluation time (cpu seconds)','classified instances','classifications correct (percent)','Kappa Statistic (percent)','Kappa Temporal Statistic (percent)','Kappa M Statistic (percent)','Pool Size','system_concept','change_detected','memory']
    store_filename_all = f"{parent / name_stem}_result_{version}_True.pickle"
    store_filename_cropped = f"{parent / name_stem}_result_{version}_False.pickle"
    print(f"attempting to read {store_filename_all}")
    try_files = [store_filename_all, store_filename_cropped]
    if readall:
        try_files = [store_filename_all]

    if any([os.path.exists(x) for x in try_files]):
        for x in try_files:
            if os.path.exists(x):
                try:
                    with open(x, "rb") as f:
                        result = pickle.load(f)
                    print(result)
                    return result
                except Exception as e:
                    print(e)
                    print("Cant read storage file")
    else:
        print("storage file does not exist")
    try:
        if not readall:
            with open(file, 'r') as f_ob:
                lastLines = tl.tail(f_ob,5)[1:]
                print(lastLines)
            print('\n'.join(lastLines))
            data = pd.read_csv(io.StringIO('\n'.join(lastLines)), header=None)
            data.columns = row_names
            print(data.head())
        else:
            data = pd.read_csv(file, header = 0)
            print(data.head())
    except Exception as e:
        print(e)
        print("no data")
        return None
    
    merges_ref = None

    if os.path.exists(merge_file):
        print('merges found')
        with open(merge_file, 'rb') as mf:
            merges_ref = pickle.load(mf)

    if 'system_concept' in data.columns and not (merges_ref is None):
        print("following merges")
        print(merges_ref)
        print(data['system_concept'])
        for i,sysc in enumerate(data['system_concept'].values):
            update_value = sysc
            while update_value in merges_ref:
                update_value = merges_ref[update_value]
            if update_value != sysc:
                data.at[i, 'system_concept'] = update_value
        print(data['system_concept'])
    print("calling csv_result_extraction")
    result = log_accuracy_from_df(data, parent / name_stem, parent, merges_ref, readall)
    link_res = link(data, aux_info, results_timestamps, aux_feature_predictions, parent)
    print(link_res)
    linked_result = {**link_res, **result}
    with open(f"{parent / name_stem}_result_{version}_{readall}.pickle", "wb") as f:
        pickle.dump(linked_result, f)
    print(f"Res {linked_result}")
    return linked_result

def test_aux_feature_prediction(data_fn, aux_fn, loaded_data = None, loaded_aux = None):
    if loaded_data is None:
        dataset = pd.read_csv(data_fn)
    else:
        dataset = loaded_data
    if loaded_aux is None:
        aux = pd.read_csv(aux_fn)
    else:
        aux = loaded_aux
    aux_feature_predictions = {}
    for aux_col in [c for c in aux.columns if c != "date_time"]:
        aux_dataset = dataset
        print(aux_col)
        print(aux_dataset.shape)
        print(aux[aux_col].shape)

        if len(np.unique(aux[aux_col])) > 16:
            aux_dataset["y"] = pd.qcut(aux[aux_col], 4, labels=False, duplicates="drop").values
        else:
            aux_dataset["y"] = aux[aux_col].astype("str")
        aux_dataset = aux_dataset.dropna()
        y = aux_dataset["y"].values
        x = aux_dataset[list(dataset.drop("y", axis = 1).columns)].values
        print(aux_dataset[list(dataset.columns)].columns)
        model_scores = []
        for i in range(3):
            model = sklearn.ensemble.RandomForestClassifier(n_estimators = 10)
            scores = sklearn.model_selection.cross_val_score(model, x, y, scoring='accuracy', cv = 5)
            model_scores.append(np.mean(scores))
        model_score = np.mean(model_scores)
        aux_feature_predictions[aux_col] = model_score
    return aux_feature_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", default="cwd", type=str, help="The directory containing the experiment")
    parser.add_argument("-o", default="cwd", type=str, help="The directory containing the output")
    parser.add_argument("-ct", default = "dynse", type=str, help="Classifier type")
    parser.add_argument("-hd", action='store_true', help="If the stream file first line is a header")
    parser.add_argument("-f", default=None, type=str, help="Overwrite results filename")
    parser.add_argument("-dbl", action='store_true', help="Debug Logging")
    parser.add_argument("-tpw", type=int, default = 0, help="Train period for windowing")
    parser.add_argument("-tp", type=int, default = 0, help="Train period")
    parser.add_argument("-dr", action='store_true', help="Direction")
    parser.add_argument("-re", type=int, default = 1, help="Repeat")
    parser.add_argument("-sds", nargs="*", type=int, default = None, help="seeds")
    parser.add_argument("-app", action='store_true', help="Append")
    parser.add_argument("-syn", action='store_true', help="Is synthetic data?")

    parser.add_argument("-rwf", default = "Rangiora", type=str, help="Name of realworld dataset")
    parser.add_argument("-ti", nargs="*", type=int, default = [0], help="target index")

    parser.add_argument("-bp", type=float, default = 0.02, help="Broken Proportion")
    parser.add_argument("-bl", type=int, default = 75, help="Broken length")
    parser.add_argument("-bcs", nargs="*", type=str, default = ["all"], help="BT sustain sensitivity")
    parser.add_argument("-bco", action='store_true', help="only run baselines")
    parser.add_argument("-rsp", action='store_true', help="remake spacial pattern")
    # parser.add_argument("-lf", default=f"experiment-{time.time()}.log", type=str, help="The name of the file to log to")

     
    args = parser.parse_args()
    # Set global args
    directory = args.d
    dir_path = pathlib.Path(directory) if directory != 'cwd' else pathlib.Path.cwd()
    header = args.hd
    print(header)
    results_filename = args.f
    train_only_period_window = args.tpw
    train_only_period = args.tp
    real_world_dataset_name = args.rwf
    real_world_target_indexs = args.ti
    using_synthetic_data = args.syn
    direction = args.dr
    seeds = args.sds
    broken_proportion = args.bp
    broken_length = args.bl
    bl_only = args.bco
    remake_spacial_pattern = args.rsp
    if args.dbl:
        logging.getLogger().setLevel(logging.DEBUG)

    O = args.o
    O_path = pathlib.Path(O) if O != 'cwd' else pathlib.Path.cwd()
    seed_counter = 0
    if args.app:
        output_dir_base = str(O_path / 'experiments' / real_world_dataset_name)
        stream_files = get_stream_files(str(output_dir_base))
        np.random.shuffle(stream_files)
        for stream_file in stream_files:

            stream_file = pathlib.Path(stream_file)
            try:
                seed = int(stream_file.parent.stem)
            except:
                seed = seed_counter
                seed_counter += 1
            ti = int(stream_file.parent.parent.stem)

            if seeds is not None:
                if seed not in seeds:
                    continue
            if real_world_target_indexs is not None:
                if ti not in real_world_target_indexs:
                    continue

            real_world_target_index = int(stream_file.parent.parent.stem)
            output_dir = stream_file.parent

            if not using_synthetic_data:
                data_fn = output_dir / f"stream-{real_world_dataset_name}_dataset.csv"
            else:
                data_fn = output_dir / f"save_data.csv"
            aux_fn = f"{output_dir / real_world_dataset_name}_aux.csv"
            info_fn = output_dir / f"df_info.json"
            time_index_fn = output_dir / f"time_index.pickle"
            dist_fn = output_dir / f"dist.json"

            experiments_dir = pathlib.Path(output_dir_base) / real_world_dataset_name
            aux_feature_preds_fn = pathlib.Path(output_dir_base) / "aux_feature_preds.json"

            if not using_synthetic_data:
                aux_feature_predictions = json.load(aux_feature_preds_fn.open())
                aux_info = pd.read_csv(aux_fn)
                results_timestamps = pd.read_pickle(time_index_fn)
                gts=[]
            else:
                dataset = pd.read_csv(data_fn)
                num_observations = dataset.shape[0]
                synthetic_timestamps = pd.date_range('1/1/2019', freq='1min', periods=num_observations)
                options_fn = output_dir / f"options.json"
                options = json.load(options_fn.open())
                gts = []
                for observation_id in range(num_observations):
                    for seg in options['boundaries']:
                        if seg[1] <= observation_id < seg[2]:
                            gt = seg[0]
                            break
                    gts.append(gt)
                aux_info = pd.DataFrame(data=gts, index=synthetic_timestamps)
                aux_info = aux_info.reset_index()
                aux_info.columns = ['date_time', 'synthetic_gt']
                print(aux_info.head())
                aux_feature_predictions = test_aux_feature_prediction(None, None, loaded_data = dataset, loaded_aux = aux_info)
                results_timestamps = aux_info['date_time']

            experiment_results_fn = output_dir / f"dataset_target_results.json"
            dataset_results_fn = pathlib.Path(output_dir_base) / f"dataset_results.json"

            results = gridsearch_experiment(output_dir, header, f'sys-{args.ct}', args.ct, train_only_period, real_world_dataset_name, real_world_target_index, aux_feature_predictions, seed, using_synthetic_data=using_synthetic_data, passed_aux=aux_info,gts=gts)

            for ro in results:
                keys = list(results[ro].keys())

            existing_experiment_results = {}
            existing_dataset_results = {}

            for r in range(1000):
                try:
                    if experiment_results_fn.exists():
                        existing_experiment_results = json.load(experiment_results_fn.open())
                    with open(str(experiment_results_fn), 'w') as f:
                        json.dump({**existing_experiment_results, **results}, f)
                    break
                except:
                    time.sleep(1)
                    continue

            for r in range(1000):
                try:
                    if experiment_results_fn.exists():
                        existing_experiment_results = json.load(experiment_results_fn.open())
                    with open(str(experiment_results_fn), 'w') as f:
                        json.dump({**existing_experiment_results, **results}, f)
                    break
                except:
                    time.sleep(1)
                    continue
    else:
        for repeat in range(args.re):
            for real_world_target_index in real_world_target_indexs:
                if seeds is None:
                    seeds = []
                if repeat >= len(seeds):
                    seed = np.random.randint(10000)
                else:
                    seed = seeds[r]
                succeded = False
                try_count = 0
                output_dir_base = str(O_path / 'experiments')
                output_dir, data_fn, aux_fn, info_fn, time_index_fn = create_dataset(dir_path / "RawData" / real_world_dataset_name, real_world_dataset_name, real_world_target_index, output_dir_base, direction=direction, seed = seed, broken_proportion = broken_proportion, broken_length = broken_length)

                aux_feature_path = pathlib.Path(output_dir_base) / real_world_dataset_name / "aux_feature_preds.json"
                try:
                    aux_feature_predictions = json.load(aux_feature_path.open())
                except:
                    aux_feature_predictions = test_aux_feature_prediction(data_fn, aux_fn)
                    with open(aux_feature_path, 'w') as f:
                        json.dump(aux_feature_predictions, f)


                experiment_results_fn = output_dir / f"dataset_target_results.json"
                dataset_results_fn = pathlib.Path(output_dir_base) / real_world_dataset_name / f"dataset_results.json"

                existing_experiment_results = {}
                existing_dataset_results = {}

                if experiment_results_fn.exists():
                    existing_experiment_results = json.load(experiment_results_fn.open())
                if dataset_results_fn.exists():
                    existing_dataset_results = json.load(dataset_results_fn.open())


                print(time_index_fn)
                aux_info = pd.read_csv(aux_fn)
                results_timestamps = pd.read_pickle(time_index_fn)
                output_dir_str = str(output_dir)
                baselines = [
                    {
                        "name": "lin",
                        "ct": "linear",
                        "train_only_period": 0
                    },
                    {
                        "name": "normSCG",
                        "ct": "normSCG",
                        "train_only_period": 0
                    },
                    {
                        "name": "temp",
                        "ct": "temporal",
                        "train_only_period": 0
                    },
                    {
                        "name": "arf",
                        "ct": "arf",
                        "train_only_period": 0
                    },
                    {
                        "name": "OK",
                        "ct": "OK",
                        "train_only_period": 0
                    },
                    {
                        "name": "tree",
                        "ct": "tree",
                        "train_only_period": int(train_only_period_window)
                    },
                ]
                if args.bcs[0] != 'all':
                    baselines = [e for e in baselines if e['name'] in args.bcs or e['ct'] in args.bcs]
                for b in baselines:
                    b['run_options'] = RunOptions(output_dir_str, header, b["name"], False, False, -1 , -1, -1, -1, -1, -1, -1, -1, -1, -1, b["ct"], 0)


                stream_files = get_stream_files(output_dir_str)
                print(stream_files)
                for stream_file in stream_files:
                    for b in baselines:
                        try:
                            fn, mg, info = run_stream_with_options(b['run_options'], stream_file, stream_file.parent, load_arff= False, train_only_period = b['train_only_period'])
                        except Exception as e:
                            raise e
                            # pass
                        r = extract_results_from_csv(fn, mg, info, stream_file.parent, True, aux_info, results_timestamps, aux_feature_predictions)
                        b['results'] = r
                        print(baselines)
                        results_dump_path = stream_file.parent / f"{b['name']}-results.json"
                        with results_dump_path.open('w') as f:
                            json.dump(r, f)


                    # Set GridSearch args
                    gridsearch_args = ['cl', 'an', 'w','sdp', 'msm', 'atl', 'atp', 'bs', 'csd', 'css']
                    gridsearch_arg_combos = product(*[vars(args)[x] for x in gridsearch_args])
                    results = gridsearch_experiment(list(gridsearch_arg_combos), gridsearch_args, output_dir, header, f'sys-{args.ct}', backtrack, proactive_sensitivity, args.ct, train_only_period, real_world_dataset_name, real_world_target_index, aux_feature_predictions, seed)

                    for ro in results:
                        keys = list(results[ro].keys())
                        for c in baselines:
                            
                            for k in keys:
                                try:
                                    if not isinstance(results[ro][k],dict):
                                        results[ro][f"{c['name']}-{k}-{'diff'}"] = results[ro][k] - c['results'][k]
                                    else:
                                        results[ro][f"{c['name']}-{k}-{'ddiff'}"] = results[ro][k]["dummy_increase"] - c['results'][k]["dummy_increase"]
                                        results[ro][f"{c['name']}-{k}-{'diff'}"] = results[ro][k]["feature_increase"] - c['results'][k]["feature_increase"]
                                except:
                                    pass
                    for b in baselines:
                        keys = list(b['results'].keys())
                        for c in baselines:
                            if b is c:
                                continue
                            for k in keys:
                                try:
                                    if not isinstance(b['results'][k],dict):
                                        b['results'][f"{c['name']}-{k}-{'diff'}"] = b['results'][k] - c['results'][k]
                                    else:
                                        b['results'][f"{c['name']}-{k}-{'ddiff'}"] = b['results'][k]["dummy_increase"] - c['results'][k]["dummy_increase"]
                                        b['results'][f"{c['name']}-{k}-{'diff'}"] = b['results'][k]["feature_increase"] - c['results'][k]["feature_increase"]
                                except:
                                    pass    
                        
                        results[json.dumps({**b["run_options"]._asdict(), "dataset_name": real_world_dataset_name, "dataset_target_index": real_world_target_index})] = b["results"]
                    with open(str(experiment_results_fn), 'w') as f:
                        json.dump({**existing_experiment_results, **results}, f)
                    with open(str(dataset_results_fn), 'w') as f:
                        json.dump({**existing_dataset_results, **results}, f)

                    gridsearch_args = ['cl', 'an', 'w','sdp', 'msm', 'atl', 'atp', 'bs', 'csd', 'css']
                    non_bt_args = {'cl': args.cl, 'an': [5], 'w': args.w, 'sdp': [500], 'msm': [1.5], 'atl': [1], 'atp': [2000], 'bs': args.bs, 'csd': [0.005], 'css': [0.1]}
                    gridsearch_arg_combos = product(*[non_bt_args[x] for x in gridsearch_args])
                    results = gridsearch_experiment(list(gridsearch_arg_combos), gridsearch_args, output_dir, header, f'sys-{args.ct}', False, False, args.ct, train_only_period, real_world_dataset_name, real_world_target_index, aux_feature_predictions, seed)

                    for ro in results:
                        keys = list(results[ro].keys())
                        for c in baselines:
                            
                            for k in keys:
                                try:
                                    if not isinstance(results[ro][k],dict):
                                        results[ro][f"{c['name']}-{k}-{'diff'}"] = results[ro][k] - c['results'][k]
                                    else:
                                        results[ro][f"{c['name']}-{k}-{'ddiff'}"] = results[ro][k]["dummy_increase"] - c['results'][k]["dummy_increase"]
                                        results[ro][f"{c['name']}-{k}-{'diff'}"] = results[ro][k]["feature_increase"] - c['results'][k]["feature_increase"]
                                except:
                                    pass
                    for b in baselines:
                        keys = list(b['results'].keys())
                        for c in baselines:
                            if b is c:
                                continue
                            for k in keys:
                                try:
                                    if not isinstance(b['results'][k],dict):
                                        b['results'][f"{c['name']}-{k}-{'diff'}"] = b['results'][k] - c['results'][k]
                                    else:
                                        b['results'][f"{c['name']}-{k}-{'ddiff'}"] = b['results'][k]["dummy_increase"] - c['results'][k]["dummy_increase"]
                                        b['results'][f"{c['name']}-{k}-{'diff'}"] = b['results'][k]["feature_increase"] - c['results'][k]["feature_increase"]
                                except:
                                    pass
                                    

                    with open(str(experiment_results_fn), 'w') as f:
                        json.dump({**existing_experiment_results, **results}, f)
                    with open(str(dataset_results_fn), 'w') as f:
                        json.dump({**existing_dataset_results, **results}, f)
    



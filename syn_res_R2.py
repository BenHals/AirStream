#%%
import pathlib
import json
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd

def get_stats(predictions, labels, print_output=False):
    if len(predictions) < 1:
        return 0, 0, 0, 0, 0, 0
    accuracy = sum(predictions ==
                   labels) / len(predictions)
    k_temporal_acc = 0
    k_majority_acc = 0
    gt_counts = Counter()
    our_counts = Counter()
    majority_guess = labels[0]
    temporal_guess = labels[0]
    kt_sum = 0
    km_sum = 0
    for o in zip(predictions, labels):
        p = o[0]
        gt = o[1]
        if gt == temporal_guess:
            k_temporal_acc += 1
            if p == gt:
                kt_sum += 1
        if gt == majority_guess:
            k_majority_acc += 1
            if p == gt:
                km_sum += 1
        gt_counts[gt] += 1
        our_counts[p] += 1

        majority_guess = gt if gt_counts[gt] > gt_counts[majority_guess] else majority_guess
        temporal_guess = gt
    kt_acc = kt_sum / k_temporal_acc
    km_acc = km_sum / k_majority_acc
    k_temporal_acc = min(k_temporal_acc / labels.shape[0], 0.99999)
    k_temporal_acc = (accuracy - k_temporal_acc) / (1 - k_temporal_acc)
    k_majority_acc = min(k_majority_acc / labels.shape[0], 0.99999)

    k_majority_acc = (accuracy - k_majority_acc) / (1 - k_majority_acc)
    expected_accuracy = 0
    for cat in np.unique(predictions):
        expected_accuracy += min((gt_counts[cat] / labels.shape[0]) *
                                  (our_counts[cat] / labels.shape[0]), 0.99999)
    if print_output:
        print(accuracy, expected_accuracy)
    k_s = (accuracy - expected_accuracy) / (1 - expected_accuracy)

    return accuracy, k_temporal_acc, k_majority_acc, k_s, kt_acc, km_acc

# directory = 'synthetic_tests'
directory = r"synthetic_tests_R2"
# base_directory = pathlib.Path.cwd() 
base_directory = pathlib.Path(r"H:\PhD\Paper2-AdaptionRepair")
directory_path = base_directory / directory
print(directory_path)

results = list(directory_path.rglob('**/*_results.pickle'))
print(results)
option_paths = directory_path.rglob('**/*options')

rows = []
count = set()
for res_path in results:
    print_output = False
    if 'dynse' in str(res_path):
        print_output = True
    container = res_path.parent
    if container not in count:
        count.add(container)
    alg = res_path.stem.split('_results')[0]
    options_path = container / 'options.json'
    with options_path.open('r') as f:
        options = json.load(f)
    with res_path.open('rb') as f:
        try:
            res = pickle.load(f)
        except:
            print(res_path)
            continue
    
    options['algorithm'] = alg
    acc, kt, km, ks, ktc, kmc = get_stats(np.array(res['predictions']), np.array(res['labels']), print_output=print_output)
    options['acc'] = acc
    options['kt'] = kt
    options['ktc'] = ktc
    options['km'] = km
    options['kmc'] = kmc
    options['ks'] = ks
    if print_output:
        print(kt, km, ks)
    # options['accuracy'] = res['accuracy']
    options['time'] = res['time']
    options['mem'] = res['mem']
    for k in res['overall_results']:
        options[k] = res['overall_results'][k]


    gts = np.array(res['gts'])
    drift_points = np.where(gts[:-1] != gts[1:])[0]
    follow_length = 150
    min_follow_period = 0
    following_drift = np.unique(np.concatenate([np.arange(i + min_follow_period, min(i+min_follow_period+follow_length+1, gts.shape[0])) for i in drift_points]))
    # filtered = data.iloc[following_drift]
    predictions = np.array(res['predictions'])[following_drift]
    labels = np.array(res['labels'])[following_drift]
    correct = (predictions == labels).astype(int)
    acc, kt, km, ks, ktc, kmc = get_stats(predictions, labels, print_output=print_output)
    options['acc_detect'] = acc
    options['kt_detect'] = kt
    options['ktc_detect'] = ktc
    options['km_detect'] = km
    options['kmc_detect'] = kmc
    options['ks_detect'] = ks
    if print_output:
        print(kt, km, ks)
    gts = np.array(res['gts'])
    drift_points = np.where(gts[:-1] != gts[1:])[0]
    follow_length = 150
    min_follow_period = 0
    following_drift = np.unique(np.concatenate([np.arange(i + min_follow_period, min(i+min_follow_period+follow_length+1, gts.shape[0])) for i in drift_points]))
    if alg != 'RF':
        following_drift = following_drift[following_drift>9000]
    # filtered = data.iloc[following_drift]
    predictions = np.array(res['predictions'])[following_drift]
    labels = np.array(res['labels'])[following_drift]
    correct = (predictions == labels).astype(int)
    acc, kt, km, ks, ktc, kmc = get_stats(predictions, labels)
    options['acc_detect_hd'] = acc
    options['kt_detect_hd'] = kt
    options['ktc_detect_hd'] = ktc
    options['km_detect_hd'] = km
    options['kmc_detect_hd'] = kmc
    options['ks_detect_hd'] = ks
    options['detect_accuracy_250_hd'] = np.sum(correct) / correct.shape[0]
    rows.append(options)

df = pd.DataFrame(rows)
df

#%%
table_items = ['ks_detect', 'ks_detect_hd', 'f1', 'time', 'mem']
table_items = ['ks_detect', 'ks_detect_hd', 'f1', 'time']
df_gb = df.groupby(['generator', 'algorithm']).agg(['mean', 'std'])
# df_res = df_gb[['accuracy', 'acc', 'km', 'kmc', 'kt', 'ktc', 'ks', 'f1', 'f1 by System', 'detect_accuracy_250', 'acc_detect', 'kt_detect', 'km_detect', 'ktc_detect', 'kmc_detect', 'ks_detect', 'time', 'mem']]
df_res = df_gb[table_items] * 100
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option("display.precision", 2)
df_res.to_pickle("syn_res.pickle")
df_res['ks_detect'] = df_res['ks_detect'] * 100
df_res['ks_detect_hd'] = df_res['ks_detect_hd'] * 100
df_res.unstack('generator')
df_stack = df_res.stack(0)
df_stack['res'] = df_stack['mean'].apply(lambda x : f"{x/100:.2f}").astype(str) + ' ('+  df_stack['std'].apply(lambda x : f"{x/100:.2f}").astype(str) + ')'
df_2 = df_stack['res'].unstack('generator')
df_2 = df_2.unstack(1)
df_2 = df_2.reindex(['NC', 'OK', 'RF', 'ARF', 'AirStreamNoBacktrack', 'AirStreamBacktrack', 'sys-dynse-0'])
df_2 = df_2.reindex(table_items, axis = 1, level = 1)
df_2.to_latex("syn_res_R2.txt")
print(df_2.head(20))
print(len(count))


# %%
df_gb
# %%
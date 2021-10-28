#%%
import pathlib
import json
import pickle
import time
from collections import Counter
import tqdm

import numpy as np
import pandas as pd

def get_stats(predictions, labels):
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
    k_s = (accuracy - expected_accuracy) / (1 - expected_accuracy)

    return accuracy, k_temporal_acc, k_majority_acc, k_s, kt_acc, km_acc

directory = 'synthetic_tests_ABC'
directory_path = pathlib.Path.cwd() / directory

results = list(directory_path.glob('**/*_results.pickle'))
option_paths = directory_path.glob('**/*options')

rows = []
count = set()
for res_path in tqdm.tqdm(results):
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
    # if alg =='RF' and 'ARF' not in alg:
    #     # print(list(res_path.parents))
    #     new_path = pathlib.Path(str(res_path.parents[0]).replace('_O2', '_O').replace('_f', '_O').split('/')[-1].replace('+gts$[0,', '').replace('bou$[[0+', '')) / f"{res_path.stem}.json"
    #     # print(new_path)
    #     try:
    #         with new_path.open('r') as f:
    #             new_res = json.load(f)
    #         # print(new_res.keys())

    #         for k in new_res:
    #             res[k] = new_res[k]
    #     except:
    #         continue
        # break
    # options['run'] = str(res_path.parent)
    options['run'] = f"{options['generator']}-{options['gen_seed']}-{options['seed']}"
    options['algorithm'] = alg
    acc, kt, km, ks, ktc, kmc = get_stats(np.array(res['predictions']), np.array(res['labels']))
    options['acc'] = acc
    options['kt'] = kt
    options['ktc'] = ktc
    options['km'] = km
    options['kmc'] = kmc
    options['ks'] = ks
    options['accuracy'] = res['accuracy']
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
    acc, kt, km, ks, ktc, kmc = get_stats(predictions, labels)
    options['acc_detect'] = acc
    options['kt_detect'] = kt
    options['ktc_detect'] = ktc
    options['km_detect'] = km
    options['kmc_detect'] = kmc
    options['ks_detect'] = ks
    gts = np.array(res['gts'])
    drift_points = np.where(gts[:-1] != gts[1:])[0]
    follow_length = 150
    min_follow_period = 0
    following_drift = np.unique(np.concatenate([np.arange(i + min_follow_period, min(i+min_follow_period+follow_length+1, gts.shape[0])) for i in drift_points]))
    # if alg != 'RF' and alg != 'RF4' and alg != 'RF12':
    following_drift_2 = following_drift[following_drift>2000]
    # elif alg != 'RF':
    #     following_drift = following_drift[following_drift>4000]
    # filtered = data.iloc[following_drift]
    predictions = np.array(res['predictions'])[following_drift_2]
    labels = np.array(res['labels'])[following_drift_2]
    correct = (predictions == labels).astype(int)
    acc, kt, km, ks, ktc, kmc = get_stats(predictions, labels)
    options['acc_detect_hd2'] = acc
    options['kt_detect_hd2'] = kt
    options['ktc_detect_hd2'] = ktc
    options['km_detect_hd2'] = km
    options['kmc_detect_hd2'] = kmc
    options['ks_detect_hd2'] = ks
    options['detect_accuracy_250_hd2'] = np.sum(correct) / correct.shape[0]
    following_drift_4 = following_drift[following_drift>4000]
    # elif alg != 'RF':
    #     following_drift = following_drift[following_drift>4000]
    # filtered = data.iloc[following_drift]
    # predictions = np.array(res['predictions'])[following_drift_4]
    # labels = np.array(res['labels'])[following_drift_4]
    # correct = (predictions == labels).astype(int)
    # acc, kt, km, ks, ktc, kmc = get_stats(predictions, labels)
    # options['acc_detect_hd4'] = acc
    # options['kt_detect_hd4'] = kt
    # options['ktc_detect_hd4'] = ktc
    # options['km_detect_hd4'] = km
    # options['kmc_detect_hd4'] = kmc
    # options['ks_detect_hd4'] = ks
    # options['detect_accuracy_250_hd4'] = np.sum(correct) / correct.shape[0]
    # following_drift_9 = following_drift[following_drift>9000]
    # # elif alg != 'RF':
    # #     following_drift = following_drift[following_drift>9000]
    # # filtered = data.iloc[following_drift]
    # predictions = np.array(res['predictions'])[following_drift_9]
    # labels = np.array(res['labels'])[following_drift_9]
    # correct = (predictions == labels).astype(int)
    # acc, kt, km, ks, ktc, kmc = get_stats(predictions, labels)
    # options['acc_detect_hd9'] = acc
    # options['kt_detect_hd9'] = kt
    # options['ktc_detect_hd9'] = ktc
    # options['km_detect_hd9'] = km
    # options['kmc_detect_hd9'] = kmc
    # options['ks_detect_hd9'] = ks
    # options['detect_accuracy_250_hd9'] = np.sum(correct) / correct.shape[0]
    # following_drift_12 = following_drift[following_drift>12000]
    # # elif alg != 'RF':
    # #     following_drift = following_drift[following_drift>12000]
    # # filtered = data.iloc[following_drift]
    # predictions = np.array(res['predictions'])[following_drift_12]
    # labels = np.array(res['labels'])[following_drift_12]
    # correct = (predictions == labels).astype(int)
    # acc, kt, km, ks, ktc, kmc = get_stats(predictions, labels)
    # options['acc_detect_hd12'] = acc
    # options['kt_detect_hd12'] = kt
    # options['ktc_detect_hd12'] = ktc
    # options['km_detect_hd12'] = km
    # options['kmc_detect_hd12'] = kmc
    # options['ks_detect_hd12'] = ks
    # options['detect_accuracy_250_hd12'] = np.sum(correct) / correct.shape[0]
    rows.append(options)

df = pd.DataFrame(rows)

#%%
df_gb = df.groupby(['generator', 'algorithm']).agg(['mean', 'std'])
# df_res = df_gb[['accuracy', 'acc', 'km', 'kmc', 'kt', 'ktc', 'ks', 'f1', 'f1 by System', 'detect_accuracy_250', 'acc_detect', 'kt_detect', 'km_detect', 'ktc_detect', 'kmc_detect', 'ks_detect', 'time', 'mem']]
# df_res = df_gb[['ks_detect', 'ks_detect_hd2',  'ks_detect_hd4',  'ks_detect_hd9',  'ks_detect_hd12', 'f1', 'time', 'mem']] * 100
# df_res = df_gb[['acc', 'kt', 'ktc', 'ks', 'ks_detect', 'ks_detect_hd2',  'ks_detect_hd4', 'f1', 'time', 'mem']] * 100
df_res = df_gb[['acc', 'kt', 'ktc', 'ks', 'ks_detect', 'ks_detect_hd2', 'f1', 'time', 'mem']] * 100
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option("display.precision", 2)
df_res.to_pickle("syn_res.pickle")
df_res['ks'] = df_res['ks'] * 100
df_res['acc'] = df_res['acc'] * 100
df_res['kt'] = df_res['kt'] * 100
df_res['ktc'] = df_res['ktc'] * 100
df_res['ks_detect'] = df_res['ks_detect'] * 100
df_res['ks_detect_hd2'] = df_res['ks_detect_hd2'] * 100
# df_res['ks_detect_hd4'] = df_res['ks_detect_hd4'] * 100
# df_res['ks_detect_hd9'] = df_res['ks_detect_hd9'] * 100
# df_res['ks_detect_hd12'] = df_res['ks_detect_hd12'] * 100
df_res.unstack('generator')
df_stack = df_res.stack(0)
df_stack['res'] = df_stack['mean'].apply(lambda x : f"{x/100:.2f}").astype(str) + ' ('+  df_stack['std'].apply(lambda x : f"{x/100:.2f}").astype(str) + ')'
df_2 = df_stack['res'].unstack('generator')
df_2 = df_2.unstack(1)
# df_2 = df_2.reindex(['NC', 'OK', 'RF', 'RF2', 'RF4', 'RF12', 'ARF', 'AirStreamNoBacktrack', 'AirStreamBacktrack'])
# df_2 = df_2.reindex(['NC', 'OK', 'RF', 'RF2', 'ARF', 'AirStreamNoBacktrack', 'AirStreamBacktrack'])
# df_2 = df_2.reindex(['NC', 'OK', 'RF', 'RF2', 'ARF', 'AirStreamNoBacktrack', 'AirStreamBacktrack', 'AirStreamBacktrackAmend'])
df_2 = df_2.reindex(['NC', 'OK', 'RF', 'RF2', 'ARF', 'AirStreamNoBacktrack', 'AirStreamBacktrackAmend'])
# df_2 = df_2.reindex(['ks_detect', 'ks_detect_hd2',  'ks_detect_hd4',  'ks_detect_hd9',  'ks_detect_hd12', 'f1', 'time', 'mem'], axis = 1, level = 1)
# df_2 = df_2.reindex(['acc', 'kt', 'ktc','ks', 'ks_detect', 'ks_detect_hd2',  'ks_detect_hd4', 'f1', 'time', 'mem'], axis = 1, level = 1)
# df_2 = df_2.reindex(['acc', 'kt', 'ktc','ks', 'ks_detect', 'ks_detect_hd2', 'f1', 'time', 'mem'], axis = 1, level = 1)
df_2 = df_2.reindex(['ks_detect', 'f1', 'time', 'mem'], axis = 1, level = 1)
df_2.to_latex("syn_res_2.txt")
print(df_2.head(20))
print(len(count))

#%%
# df

# %%

# %%
# df_g

# %%
pd.set_option('display.max_rows', 2000)
df_g = df.groupby(['generator', 'run', 'algorithm']).agg('mean')
measure = 'ks_detect'
# measure = 'f1'
# measure = 'time'
# measure = 'mem'
dataset = 'reverse_RBF'
dataset = 'reverse_tree'
for dataset in ['reverse_tree', 'reverse_RBF']:
    for measure in ['ks_detect', 'f1', 'time', 'mem']:
        systems_to_compare = [x for x in df['algorithm'].unique() if x not in ["RF2", 'AirStreamBacktrack']]
        df_rank = df_g[[measure]]

        df_rank_long = df_rank.reset_index()
        df_rank_long = df_rank_long.loc[df_rank_long['algorithm'].isin(systems_to_compare)]
        df_rank_long['ranking'] = df_rank_long.groupby('run')[measure].rank(ascending=False)
        df_rank_gb = df_rank_long.groupby(['generator', 'run', 'algorithm']).mean()
        df_rank_u = df_rank_gb.unstack('algorithm')
        r = df_rank_u.xs(dataset)
        from Orange.evaluation import compute_CD, graph_ranks
        means = []
        for system in systems_to_compare:
            col = r.loc[:, ('ranking', system)]
            means.append(col.mean())
        print(means)
        cd = compute_CD(means, r.shape[0], test = 'bonferroni-dunn')
        print(cd)
        graph_ranks(means, systems_to_compare, cd, width = 10, textspace = 3, filename = f"{dataset}-{measure}")

# %%
r.loc[:, ('ks_detect_hd2', ('AirStreamBacktrack', 'AirStreamNoBacktrack', 'ARF', 'RF'))].mean()

# %%

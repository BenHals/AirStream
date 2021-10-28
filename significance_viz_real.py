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

# directory = r'S:\PhD\AirStream'
directory = r'H:\PhD\Paper2-AdaptionRepair\ReturnScaleFin'
test_name = r"experiments"
directory_path = pathlib.Path(directory) / test_name

results = list(directory_path.glob('**/*-results.json'))
# option_paths = directory_path.glob('**/*options')

rows = []
count = set()
for res_path in tqdm.tqdm(results):
    container = res_path.parent
    if container not in count:
        count.add(container)
    alg = res_path.stem.split('-results')[0]
    options_path = container / f'{alg}_run_options.txt'
    with options_path.open('r') as f:
        options = json.load(f)
    with res_path.open('rb') as f:
        try:
            res = json.load(f)
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
    print(options)
    file_parents = str(res_path).split('\\')
    print(file_parents)
    seed = file_parents[-2]
    ti = file_parents[-3]
    generator = file_parents[-4]
    options['generator'] = generator
    options['seed'] = seed
    options['gen_seed'] = ti
    options['run'] = f"{options['generator']}-{options['gen_seed']}-{options['seed']}"
    alg = options['ct'] + (str(options['backtrack']) if 'backtrack' in options else "False")
    options['algorithm'] = alg
    rows.append(options)

df = pd.DataFrame(rows)

#%%
df_gb = df.groupby(['generator', 'algorithm']).agg(['mean', 'std'])
# df_res = df_gb[['accuracy', 'acc', 'km', 'kmc', 'kt', 'ktc', 'ks', 'f1', 'f1 by System', 'detect_accuracy_250', 'acc_detect', 'kt_detect', 'km_detect', 'ktc_detect', 'kmc_detect', 'ks_detect', 'time', 'mem']]
# df_res = df_gb[['ks_detect', 'ks_detect_hd2',  'ks_detect_hd4',  'ks_detect_hd9',  'ks_detect_hd12', 'f1', 'time', 'mem']] * 100
# df_res = df_gb[['acc', 'kt', 'ktc', 'ks', 'ks_detect', 'ks_detect_hd2',  'ks_detect_hd4', 'f1', 'time', 'mem']] * 100
df_res = df_gb[['acc', 'kt', 'ktc', 'ks']] * 100
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option("display.precision", 2)
df_res.to_pickle("syn_res.pickle")
df_res['ks'] = df_res['ks'] * 100
df_res['acc'] = df_res['acc'] * 100
df_res['kt'] = df_res['kt'] * 100
df_res['ktc'] = df_res['ktc'] * 100
# df_res['ks_detect_hd4'] = df_res['ks_detect_hd4'] * 100
# df_res['ks_detect_hd9'] = df_res['ks_detect_hd9'] * 100
# df_res['ks_detect_hd12'] = df_res['ks_detect_hd12'] * 100
df_res.unstack('generator')
df_stack = df_res.stack(0)
df_stack['res'] = df_stack['mean'].apply(lambda x : f"{x/100:.2f}").astype(str) + ' ('+  df_stack['std'].apply(lambda x : f"{x/100:.2f}").astype(str) + ')'
df_2 = df_stack['res'].unstack('generator')
# df_2 = df_2.unstack(1)
# df_2 = df_2.reindex(['NC', 'OK', 'RF', 'RF2', 'RF4', 'RF12', 'ARF', 'AirStreamNoBacktrack', 'AirStreamBacktrack'])
# df_2 = df_2.reindex(['NC', 'OK', 'RF', 'RF2', 'ARF', 'AirStreamNoBacktrack', 'AirStreamBacktrack'])
# df_2 = df_2.reindex(['NC', 'OK', 'RF', 'RF2', 'ARF', 'AirStreamNoBacktrack', 'AirStreamBacktrack', 'AirStreamBacktrackAmend'])
# df_2 = df_2.reindex(['NC', 'OK', 'RF', 'RF2', 'ARF', 'AirStreamNoBacktrack', 'AirStreamBacktrack'])
# df_2 = df_2.reindex(['ks_detect', 'ks_detect_hd2',  'ks_detect_hd4',  'ks_detect_hd9',  'ks_detect_hd12', 'f1', 'time', 'mem'], axis = 1, level = 1)
# df_2 = df_2.reindex(['acc', 'kt', 'ktc','ks', 'ks_detect', 'ks_detect_hd2',  'ks_detect_hd4', 'f1', 'time', 'mem'], axis = 1, level = 1)
# df_2 = df_2.reindex(['acc', 'kt', 'ktc','ks', 'ks_detect', 'ks_detect_hd2', 'f1', 'time', 'mem'], axis = 1, level = 1)
# df_2 = df_2.reindex(['ks_detect', 'f1', 'time', 'mem'], axis = 1, level = 1)

#%%
df_2

# %%
df_gb

# %%
# df_g

# %%
pd.set_option('display.max_rows', 2000)
df_g = df.groupby(['generator', 'run', 'algorithm']).agg('mean')
for dataset in ['RangioraClean', 'Arrowtown', 'BMSClean']:
    for measure in ['ks']:
        systems_to_compare = [x for x in df['algorithm'].unique() if x not in ["RF2"]]
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
        means = [x if not np.isnan(x) else 0 for x in means]
        print(means)
        print(r.shape[0])
        # cd = compute_CD(means, r.shape[0], test = 'bonferroni-dunn')
        cd = compute_CD(means, r.shape[0])
        print(cd)
        graph_ranks(means, systems_to_compare, cd, width = 10, textspace = 3, filename = f"{dataset}-{measure}_v2")

# %%
r.loc[:, ('ks_detect_hd2', ('AirStreamBacktrack', 'AirStreamNoBacktrack', 'ARF', 'RF'))].mean()

# %%

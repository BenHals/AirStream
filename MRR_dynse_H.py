#%%
import pathlib
import json
import tqdm
from collections import Counter

import numpy as np
import pandas as pd

# base_directory = pathlib.Path.cwd() / "ReturnScaleFin" / "experiments"
# base_directory = pathlib.Path.cwd() / "ReturnScaleFin" / "experiments" / "RangioraClean"
base_directory = pathlib.Path(r"H:\PhD\Paper2-AdaptionRepair\RepairResearchData\clean_experiments_no_sys_dynse\experiments") / "RangioraClean"
base_directory = pathlib.Path(r"S:\PhD\AirStream\experiments") / "RangioraClean"

# base_directory = pathlib.Path.cwd() / "Ret" / "experiments"

options_files = list(base_directory.glob('**/*_run_options.txt*'))
def get_stats(predictions, labels):
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
#%%
results = []
for option_path in tqdm.tqdm(options_files):
    try:
        with option_path.open('r') as f:
            options = json.load(f)
    except:
        print("except")
        continue
    options_path_n = option_path.parent / f"{option_path.stem}_n.json"
    prefix = option_path.stem.split('_run_options')[0]
    result_path = option_path.parent / f"{prefix}.csv"
    result_json_path = option_path.parent / f"{prefix}-results.json"
    result_json_path2 = option_path.parent / f"{prefix}-results.json"
    stats_path = option_path.parent / f"{prefix}-run_stats.txt"
    ti = option_path.parents[1].stem
    rwf = option_path.parents[2].stem
    options['run'] = str(result_path.parent)
    options['sys'] = options['filename'].split('-')[0]
    options['algorithm'] = options['ct']
    print(options['algorithm'])
    if 'backtrack' not in options:
        options['backtrack'] = "-1"
    if 'proactive_sensitivity' not in options:
        options['proactive_sensitivity'] = "-1"
    if 'cl' not in options:
        options['cl'] = -1
    if 'an' not in options:
        options['an'] = -1
    if 'window' not in options:
        options['window'] = -1
    if 'sdp' not in options:
        options['sdp'] = -1
    if 'msm' not in options:
        options['msm'] = -1
    if 'atl' not in options:
        options['atl'] = -1
    if 'atp' not in options:
        options['atp'] = -1
    if 'bs' not in options:
        options['bs'] = -1
    if 'csd' not in options:
        options['csd'] = -1
    if 'css' not in options:
        options['css'] = -1
    if options['ct'] == 'fsm':
        options['algorithm'] = f"AirStream-Backtrack:{options['backtrack']}"
    options['system'] = options['sys'] + options['algorithm'] + str(options['window'])
    # print(options['system'])
    # if options['system'] not in ['OKOK-1', 'arfarf-1', 'linlinear-1', 'normSCGnormSCG-1', 'temptemporal-1', 'treetree-1', 'sysAirStream-Backtrack:True1500', 'sysAirStream-Backtrack:False1500']:
        # print(f"{options['system']} is not in")
        # continue
    # if options['sys'] != 'ds1_bt_amend':
    #     continue
    # recheck = True
    recheck = False
    # print('time' not in options)
    if 'time' not in options or recheck:
        found = False
        if options_path_n.exists() and (not recheck):
            with options_path_n.open('r') as f:
                options_json = json.load(f)
                if 'algorithm' in options_json:
                    found = True
                    options = options_json
        if result_path.exists() and (not found or 'time' not in options):
            # data = pd.read_csv(result_path, names = ["observation", "SW_accuracy", "is_correct", "p", "y", "Accuracy", "active_state_id", "found_change", "set_restore_state", "restore_state", "signal_confidence_backtrack", "signal_difference_backtrack", "current_sensitivity", "mask"])
            
            print(result_path)
            print(options)
            if options['algorithm'] == 'dynse':
                data = pd.read_csv(result_path, header=0)
            else:
                # data = pd.read_csv(result_path, names = ["example", "sliding_window_accuracy", "is_correct", "p", "y", "Accuracy", "active_state_id", "found_change", "set_restore_state", "restore_state", "signal_confidence_backtrack", "signal_difference_backtrack", "current_sensitivity", "mask"])
                data = pd.read_csv(result_path)
                if len(data.columns) == 14:
                    data.columns = ["example", "sliding_window_accuracy", "is_correct", "p", "y", "Accuracy", "active_state_id", "found_change", "set_restore_state", "restore_state", "signal_confidence_backtrack", "signal_difference_backtrack", "current_sensitivity", "mask"]
                else:
                    data.columns = ["example", "sliding_window_accuracy", "is_correct", "p", "y", "Accuracy", "active_state_id", "as_id_2", "found_change", "set_restore_state", "restore_state", "signal_confidence_backtrack", "signal_difference_backtrack", "current_sensitivity", "mask"]
            print(data.head())
            options['clean_target'] = True
            if 'Rangiora' in rwf:
                if ti in [3, 7, 10]:
                    options['clean_target'] = False
            predictions = data['p'].astype(int)
            labels = data['y']
            acc, kt, km, ks, ktc, kmc = get_stats(predictions.values, labels.values)
            options['acc'] = acc
            options['kt'] = kt
            options['ktc'] = ktc
            options['km'] = km
            options['kmc'] = kmc
            options['ks'] = ks
            joined = pd.concat([predictions, labels], axis=1)
            joined['diff'] = joined['p'] - joined['y']
            data['ranking'] = predictions - labels
            data['ranking'] = data['ranking'].abs()
            data['ranking'] = data['ranking'] + 1
            data['ranking'].value_counts()
            data['reciprocal_ranking'] = 1 / data['ranking']
            data['reciprocal_ranking'].value_counts()
            mean_reciprocal_rank = data['reciprocal_ranking'].mean()
            options['mean_reciprocal_rank'] = mean_reciprocal_rank
            options['accuracy'] = (predictions == labels).sum() / predictions.shape[0]
            options['run'] = str(result_path.parent)
            # print("past init")
            try:
                print(stats_path)
                with stats_path.open('r') as f:
                    time = float(f.read().split(': ')[1])
                print(time)
            except Exception as e:
                print(e)
                time = np.nan
            options['time'] = time

            levels = [int(x) for x in labels.unique()]
            # print(levels)
            pred_levels = labels.unique()
            # print(pred_levels)
            conf_matrix = {}
            for l in levels:
                row = joined.loc[joined['y'] == l]
                conf_matrix[l] = {}
                for p in levels:
                    col = row.loc[row['p'] == p]
                    conf_matrix[l][p] = int(col.shape[0])
            options['conf_matrix'] = conf_matrix
            # print("past conf init")
            for level in levels:
                level_rows = joined.loc[joined['y'] == level]
                data[f'{level}_ranking'] = level_rows['p'] - level_rows['y']
                data[f'{level}_ranking'] = data[f'{level}_ranking'].abs()
                data[f'{level}_ranking'] = data[f'{level}_ranking'] + 1
                data[f'{level}_ranking'].value_counts()
                data[f'{level}_reciprocal_ranking'] = 1 / data[f'{level}_ranking']
                data[f'{level}_reciprocal_ranking'].value_counts()
                mean_reciprocal_rank = data[f'{level}_reciprocal_ranking'].mean()
                options[f'{level}_mean_reciprocal_rank'] = mean_reciprocal_rank
                options[f'{level}_accuracy'] = (level_rows['p'] == level_rows['y']).sum() / level_rows.shape[0]
                # Recall is proportion correct over total for that label, i.e. prop of label correct
                label_sum = sum([conf_matrix[level][p] for p in levels])
                options[f'{level}_recall'] = (conf_matrix[level][level] / label_sum) if label_sum != 0 else 0
                # Precision is proportion correct over total for that prediction, i.e. prop of pred correct
                pred_sum = sum([conf_matrix[p][level] for p in levels])
                options[f'{level}_precision'] = (conf_matrix[level][level] / pred_sum) if pred_sum != 0 else 0
            # print("past ranking")
            if 'tree' not in options['ct']:
                predictions_ho = data.iloc[20000:]['p']
                # predictions_ho = data.iloc[30000:]['p']
                # print(predictions_ho.shape)
                # print(predictions.shape)
                labels_ho = data.iloc[20000:]['y']
                # labels_ho = data.iloc[30000:]['y']
                joined_ho = joined.iloc[20000:]
                # joined_ho = joined.iloc[30000:]
            else:
                predictions_ho = data['p']
                # print(predictions_ho.shape)
                # print(predictions.shape)
                # labels_ho = data.iloc[20000:]['y']
                labels_ho = data['y']
                # joined_ho = joined.iloc[20000:]
                joined_ho = joined
            # print("past ho")
            conf_matrix = {}
            for l in levels:
                row = joined_ho.loc[joined_ho['y'] == l]
                conf_matrix[l] = {}
                for p in levels:
                    col = row.loc[row['p'] == p]
                    conf_matrix[l][p] = int(col.shape[0])
            # print(conf_matrix)
            options['conf_matrix_ho'] = conf_matrix
            data['ranking'] = predictions_ho - labels_ho
            data['ranking'] = data['ranking'].abs()
            data['ranking'] = data['ranking'] + 1
            data['ranking'].value_counts()
            data['reciprocal_ranking'] = 1 / data['ranking']
            data['reciprocal_ranking'].value_counts()
            mean_reciprocal_rank = data['reciprocal_ranking'].mean()

            options['mean_reciprocal_rank_ho'] = mean_reciprocal_rank
            options['accuracy_ho'] = (predictions_ho == labels_ho).sum() / predictions_ho.shape[0]
            acc, kt, km, ks, ktc, kmc = get_stats(predictions_ho.values, labels_ho.values)
            options['acc_ho'] = acc
            options['kt_ho'] = kt
            options['ktc_ho'] = ktc
            options['km_ho'] = km
            options['kmc_ho'] = kmc
            options['ks_ho'] = ks
            for level in levels:
                level_rows = joined_ho.loc[joined_ho['y'] == level]
                data[f'{level}_ranking'] = level_rows['p'] - level_rows['y']
                data[f'{level}_ranking'] = data[f'{level}_ranking'].abs()
                data[f'{level}_ranking'] = data[f'{level}_ranking'] + 1
                data[f'{level}_ranking'].value_counts()
                data[f'{level}_reciprocal_ranking'] = 1 / data[f'{level}_ranking']
                data[f'{level}_reciprocal_ranking'].value_counts()
                mean_reciprocal_rank = data[f'{level}_reciprocal_ranking'].mean()
                options[f'{level}_mean_reciprocal_rank_ho'] = mean_reciprocal_rank
                options[f'{level}_accuracy_ho'] = (level_rows['p'] == level_rows['y']).sum() / level_rows.shape[0]
                # Recall is proportion correct over total for that label, i.e. prop of label correct
                # label_sum = sum([conf_matrix[level][p] for p in levels])
                # options[f'{level}_recall_ho'] = (conf_matrix[level][level] / label_sum) if label_sum != 0 else 0
                # # Precision is proportion correct over total for that prediction, i.e. prop of pred correct
                # pred_sum = sum([conf_matrix[p][level] for p in levels])
                # options[f'{level}_precision_ho'] = (conf_matrix[level][level] / pred_sum) if pred_sum != 0 else 0
            try:
                with option_path.open('w') as f:
                    json.dump(options, f)
            except:
                print('except')
                for k in options:
                    print(f"{k}:{type(options[k])}")
                    if isinstance(options[k], dict):
                        for k2 in options[k]:
                            print(f"{k2}:{type(options[k][k2])}")
                exit()
            with options_path_n.open('w') as f:
                json.dump(options, f)
            
    options['ti'] = ti
    options['rwf'] = rwf
    
    dataset = [x for x in option_path.parents][2]
    options['dataset'] = dataset.stem
    options['run'] = str(result_path.parent)
    options['sys'] = options['filename'].split('-')[0]
    options['algorithm'] = options['ct']
    if options['ct'] == 'fsm':
        options['algorithm'] = f"AirStream-Backtrack:{options['backtrack']}"
    options['system'] = options['sys'] + options['algorithm'] + str(options['window'])

    results.append(options)
    # print('done')

df = pd.DataFrame(results)

#%%
print(df)
#%%
# data.head()
# data['ranking'] = data['y'] - data['p']
# data['ranking'] = data['ranking'].abs()
# data['ranking'] = data['ranking'] + 1
# data['ranking'].value_counts()
# data['reciprocal_ranking'] = 1 / data['ranking']
# data['reciprocal_ranking'].value_counts()
# mean_reciprocal_rank = data['reciprocal_ranking'].mean()
# print(mean_reciprocal_rank)
#%%
# df[.loc[df['sys'] == 'ds1_bt'].columns]

#%%
# df['ti'] = df['directory'].str.split('/', expand = True)[6]
df['sys'] = df['filename'].str.split('-', expand = True)[0]

#%%
# df_g['ti'] = df_g.get_level_values().split('/')[-2]
df_g = df.groupby(['dataset', 'system', 'backtrack', 'window', 'css', 'csd']).agg(('mean', 'std', 'count'))

#%%
# idx = pd.IndexSlice
# df_g.loc[idx[:, ['arf', 'OK', 'AirStream-Backtrack:True', 'AirStream-Backtrack:False', 'linear', 'normSCG', 'temporal'], :], 'mean_reciprocal_rank']

#%%
# df_g.loc[idx[:, ['arf', 'OK', 'AirStream-Backtrack:True', 'AirStream-Backtrack:False', 'linear', 'normSCG', 'temporal'], :], :].columns


# %%
pd.set_option('display.max_rows', 1500)
# df_g[['mean_reciprocal_rank', 'accuracy', 'mean_reciprocal_rank_ho', 'accuracy_ho', '4_recall', '0_precision_ho']]
# df_g[['ks', 'mean_reciprocal_rank', 'mean_reciprocal_rank_ho', 'accuracy', 'accuracy_ho']]
df_g[['mean_reciprocal_rank_ho', 'ks_ho', 'km_ho']]
# df_g.to_csv('MRR_dynse_Rangiora.csv')

#%%
df_g['mean_reciprocal_rank_ho']
# df_g.columns


# %%
df_g.to_latex('MRR_dynse_Rangiora.txt')

#%%
df
#%%
df_data = df[['algorithm', 'mean_reciprocal_rank_ho']]
df_data

# %%
df_t = df
df_t['system'] = df_t['sys'] + df_t['algorithm'] + df_t['window'].astype(str)
df_g = df_t.groupby(['dataset', 'run', 'system']).agg('mean')

# %%
pd.set_option('display.max_rows', 9000)
df_g['mean_reciprocal_rank_ho']

# %%
df_u = df_g['mean_reciprocal_rank_ho']
# df_u = df_u.unstack('sys')
# df_u = df_u.unstack('algorithm')
# df_u = df_u.unstack('window')
df_u = df_u.unstack('system')


# %%
df_u.columns

# %%
r = df_u.xs('RangioraClean')[['OKOK-1', 'arfarf-1', 'linlinear-1', 'temptemporal-1', 'treetree-1', 'sysdynse-1']]
r['OKOK-1'] = r['OKOK-1'].fillna(r['OKOK-1'].min())
# r = df_u.xs('RangioraClean')[['sysAirStream-Backtrack:False1500', 'sysAirStream-Backtrack:True1500']]
r = r.dropna()

# %%

# %%
r.iloc[0].values

# %%

# %%
df_u

# %%
from scipy.stats import friedmanchisquare
data_mat = [r.iloc[i].values for i in range(r.shape[0])]
# print(data_mat)
print(friedmanchisquare(*data_mat))

# %%
df_rank = df_g[['mean_reciprocal_rank_ho']]

# %%
df_t.groupby('run')['mean_reciprocal_rank_ho'].rank(ascending=False)

# %%
df_rank_long = df_rank.reset_index()
df_rank_long = df_rank_long.loc[df_rank_long['system'].isin(['OKOK-1', 'arfarf-1', 'linlinear-1', 'temptemporal-1', 'treetree-1', 'sysdynse-1'])]
df_rank_long['ranking'] = df_rank_long.groupby('run')['mean_reciprocal_rank_ho'].rank(ascending=False)

# %%
df_rank_gb= df_rank_long.groupby(['dataset', 'run', 'system']).mean()
df_rank_u = df_rank_gb.unstack('system')

# %%
# r = df_rank_u.xs('RangioraClean')[['OKOK-1', 'arfarf-1', 'linlinear-1', 'normSCGnormSCG-1', 'temptemporal-1', 'treetree-1', 'sysAirStream-Backtrack:True1500', 'sysAirStream-Backtrack:True2500', 'sysAirStream-Backtrack:False1500', 'sysAirStream-Backtrack:False2500']]
# r['OKOK-1'] = r['OKOK-1'].fillna(r['OKOK-1'].min())
systems_to_compare = ['sysAirStream-Backtrack:False1500', 'sysAirStream-Backtrack:True1500']
systems_to_compare = ['OKOK-1', 'arfarf-1', 'linlinear-1', 'normSCGnormSCG-1', 'temptemporal-1', 'treetree-1', 'sysAirStream-Backtrack:True1500', 'sysAirStream-Backtrack:True2500', 'sysAirStream-Backtrack:False1500', 'sysAirStream-Backtrack:False2500']
systems_to_compare = ['OKOK-1', 'arfarf-1', 'linlinear-1', 'normSCGnormSCG-1', 'temptemporal-1', 'treetree-1', 'sysAirStream-Backtrack:True1500', 'sysAirStream-Backtrack:False1500']
systems_to_compare = ['OKOK-1', 'arfarf-1', 'linlinear-1', 'temptemporal-1', 'treetree-1', 'sysdynse-1']
r = df_rank_u.xs('RangioraClean').loc[:, (slice(None), systems_to_compare)]
r = r.dropna()

# %%
df_rank_u.xs('RangioraClean')

# %%
r

# %%
from Orange.evaluation import compute_CD, graph_ranks

means = []
for system in systems_to_compare:
    col = r.loc[:, ('ranking', system)]
    means.append(col.mean())
print(means)

# %%
cd = compute_CD(means, r.shape[0], test = 'bonferroni-dunn')
print(cd)
graph_ranks(means, systems_to_compare, cd)

# %%
measure = 'mean_reciprocal_rank'
# systems_to_compare = ['OKOK-1', 'arfarf-1', 'linlinear-1', 'normSCGnormSCG-1', 'temptemporal-1', 'treetree-1', 'sysAirStream-Backtrack:True1500', 'sysAirStream-Backtrack:False1500']
# systems_to_compare = ['OKOK-1', 'arfarf-1', 'linlinear-1', 'normSCGnormSCG-1', 'temptemporal-1', 'sysAirStream-Backtrack:True1500', 'sysAirStream-Backtrack:False1500']
systems_to_compare = ['OKOK-1', 'arfarf-1', 'linlinear-1', 'temptemporal-1', 'sysdynse-1']
dataset = 'RangioraClean'
df_t = df
# df_t['system'] = df_t['sys'] + df_t['algorithm'] + df_t['window'].astype(str)
df_g = df_t.groupby(['dataset', 'run', 'system']).agg('mean')

df_rank = df_g[[measure]]
df_rank_long = df_rank.reset_index()
# df_rank_long = df_rank_long.loc[df_rank_long['system'].isin(['OKOK-1', 'arfarf-1', 'linlinear-1', 'normSCGnormSCG-1', 'temptemporal-1', 'treetree-1', 'sysAirStream-Backtrack:True1500', 'sysAirStream-Backtrack:True2500', 'sysAirStream-Backtrack:False1500', 'sysAirStream-Backtrack:False2500'])]
df_rank_long = df_rank_long.loc[df_rank_long['system'].isin(['OKOK-1', 'arfarf-1', 'linlinear-1', 'temptemporal-1', 'sysdynse-1'])]
df_rank_long['ranking'] = df_rank_long.groupby('run')[measure].rank(ascending=False)
df_rank_gb= df_rank_long.groupby(['dataset', 'run', 'system']).mean()
df_rank_u = df_rank_gb.unstack('system')
# r = df_rank_u.xs('RangioraClean')[['OKOK-1', 'arfarf-1', 'linlinear-1', 'normSCGnormSCG-1', 'temptemporal-1', 'treetree-1', 'sysAirStream-Backtrack:True1500', 'sysAirStream-Backtrack:True2500', 'sysAirStream-Backtrack:False1500', 'sysAirStream-Backtrack:False2500']]
# r['OKOK-1'] = r['OKOK-1'].fillna(r['OKOK-1'].min())
# systems_to_compare = ['sysAirStream-Backtrack:False1500', 'sysAirStream-Backtrack:True1500']
# systems_to_compare = ['OKOK-1', 'arfarf-1', 'linlinear-1', 'normSCGnormSCG-1', 'temptemporal-1', 'treetree-1', 'sysAirStream-Backtrack:True1500', 'sysAirStream-Backtrack:True2500', 'sysAirStream-Backtrack:False1500', 'sysAirStream-Backtrack:False2500']
r = df_rank_u.xs(dataset).loc[:, (slice(None), systems_to_compare)]
r = r.dropna()

from scipy.stats import friedmanchisquare
data_mat = [r.iloc[i].values[:len(systems_to_compare)] for i in range(r.shape[0])]
# print(data_mat)
print(friedmanchisquare(*data_mat))

from Orange.evaluation import compute_CD, graph_ranks
means = []
for system in systems_to_compare:
    col = r.loc[:, ('ranking', system)]
    means.append(col.mean())
print(means)
cd = compute_CD(means, r.shape[0], test = 'bonferroni-dunn')
# cd = compute_CD(means, r.shape[0], test = 'nemenyi')
print(cd)
graph_ranks(means, systems_to_compare, cd, width = 10, textspace = 3, filename = f"{dataset}-{measure}")

# %%
df['system'].unique()

# %%

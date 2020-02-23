#%%
import pathlib
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def get_confusion_matrix(preds, labels, levels):
    if levels is None:
        levels = set()
    yp_matches = {}
    for p,y in zip(preds, labels):
        try:
            p = int(p)
        except:
            p = int(p.replace('[', '').replace(']', ''))
        
        y = int(y)
        levels.add(p)
        levels.add(y)
        if y not in yp_matches:
            yp_matches[y] = {}
        if p not in yp_matches[y]:
            yp_matches[y][p] = 0
        yp_matches[y][p] += 1
    matrix = []
    for y in levels:
        lev = []
        matrix.append(lev)
        for p in levels:
            if y not in yp_matches:
                lev.append(0)
                continue
            if p in yp_matches[y]:
                lev.append(yp_matches[y][p])
            else:
                lev.append(0)
    return np.array(matrix).reshape(len(levels), len(levels))
#%%
# data_name = "BMSBase47"
data_name = "RangioraClean"
target_index = -1
target_seed = -1
experiment_directory = pathlib.Path.cwd() / 'experiments' / data_name
if target_index >= 0:
    experiment_directory = experiment_directory / str(target_index)
if target_seed >= 0:
    experiment_directory = experiment_directory / str(target_seed)
print(experiment_directory)
# data_name = "BMSBase47"
data_name = "RangioraClean"
#%%
parser = argparse.ArgumentParser()
parser.add_argument("-o", default="cwd", type=str, help="The directory containing the experiment")
parser.add_argument("-fs", default="rang", type=str, help="Feature set")
parser.add_argument("-rd", default="RangioraClean", type=str, help="The directory containing the experiment")
parser.add_argument("-rdr", default="experiments", type=str, help="The directory containing the experiment")
parser.add_argument("-dn", default="RangioraClean", type=str, help="The directory containing the experiment")
parser.add_argument("-ti", default=-1, type=int, help="The directory containing the experiment")
parser.add_argument("-seed", default=-1, type=int, help="The directory containing the experiment")

args = parser.parse_args()

# data_name = "Rangiora"
data_name = args.rd
target_index = args.ti
target_seed = args.seed
feature_set = args.fs
# data_name = "BMS"
origin_path = pathlib.Path.cwd() if args.o == 'cwd' else pathlib.Path(args.o)

experiment_directory = origin_path / args.rdr / data_name
if target_index >= 0:
    experiment_directory = experiment_directory / str(target_index)
if target_seed >= 0:
    experiment_directory = experiment_directory / str(target_seed)
print(experiment_directory)
data_name = args.dn

#%%
print(experiment_directory)
data_files = experiment_directory.glob('**/*dataset_target_results.json')
res = {}
for data_f in data_files:
    res = {**json.load(data_f.open()), ** res}
# print(res)
# print(len(list(res.keys())))
all_results = res
# print(all_results)
formatted_results = {}
index_names = []
for k in all_results:
    key = json.loads(k)
    if 'gitcommit' in k:
        del key['gitcommit']
    if 'dataset_target_index' in k:
        del key['dataset_target_index']
    index_names = key.keys()
    # print(len(list(index_names)))
    # print(key)
    # print(list(index_names))

    # if 'gitcommit' not in list(index_names):
    #     formatted_results[tuple([*key.values(), 'gitcommit'])] = {**all_results[k]}
    #     index_names = [*list(index_names), 'gitcommit']
    #     pointer = formatted_results[tuple([*key.values(), 'gitcommit'])]
    # else:
    formatted_results[tuple(key.values())] = {**all_results[k]}
    pointer = formatted_results[tuple(key.values())]
    pointer_keys = list(pointer.keys())
    for fk in pointer_keys:
        if type(pointer[fk]) is dict:
            keys = pointer[fk].keys()
            for sub_dict_key in keys:
                full_key = f"{fk}-{sub_dict_key}"
                pointer[full_key] = pointer[fk][sub_dict_key]
            del pointer[fk]
results = pd.DataFrame.from_dict(formatted_results, orient='index').rename_axis(index_names)
results['feature_set'] = results.index.get_level_values('ct').values
results['feature_set'] = results['feature_set'] + results.index.get_level_values('backtrack').astype('str').values
results['feature_set'] = results['feature_set'] + results.index.get_level_values('proactive_sensitivity').astype('str').values

results_files = list(experiment_directory.glob('**/*.csv'))
test_df = results.groupby(['ct', 'backtrack', 'proactive_sensitivity', 'bs', 'window']).mean()
test_df_show_cols = ['accuracy']
if 'link-WD_4-model_score' in test_df:
    test_df_show_cols.append('link-WD_4-model_score')
# print(test_df[test_df_show_cols])


# level_counts = []
# failed = set()
# for result_path in results_files:
#     print(result_path)
#     if data_name in str(result_path.stem) or 'full_link' in str(result_path.stem) or 'drift_info' in str(result_path.stem) or 'stream' in str(result_path.stem):
#         continue
#     try:
#         res = pd.read_csv(result_path, header=None)
#     except:
#         failed.add(result_path)
#     # print(res.head())
#     # print(res.head(3))
#     level_counts.append(res[3].nunique())
# # print(level_counts)
# levels = set(list(range(max(level_counts))))

failed = set()
levels = [0, 1, 2, 3, 4, 5]
if feature_set == "synth":
    levels = [0, 1]
    if data_name == "WINDSIM":
        levels = [0, 1, 2, 3, 4]
levels = set(levels)
from tqdm import tqdm
rows = []
results_files = list(experiment_directory.glob('**/*.csv'))
for result_path in tqdm(results_files):
    print(result_path)
    if result_path in failed:
        continue
    if data_name in str(result_path.stem) or 'full_link' in str(result_path.stem) or 'drift_info' in str(result_path.stem) or 'stream' in str(result_path.stem) or 'aux' in str(result_path.stem):
        continue
    
    rwf = result_path.parent.parent.parent.stem
    seed = int(result_path.parent.stem)
    try:
        ti = int(result_path.parent.parent.stem)
    except:
        ti = 0
        rwf = result_path.parent.parent.stem
    
    if 'Rangiora' in str(rwf) and ti in [3, 7, 10]:
        continue

    run_options = result_path.parent / f"{result_path.stem}_run_options.txt"
    run_options = json.load(run_options.open())
    results_json_path = result_path.parent / f"{result_path.stem}-results.json"
    try:
        results_from_json = json.load(results_json_path.open())
    except:
        results_from_json = {}
    results_json_path = result_path.parent / f"{result_path.stem}.csv-results.json"
    try:
        results_from_json = json.load(results_json_path.open())
    except:
        results_from_json = results_from_json
    try:
        aux_info = pd.read_csv(result_path.parent / f"{data_name}_aux.csv")
    except:
        aux_info = pd.read_csv(result_path.parent / f"drift_info.csv")
    try:
        dataset_info = pd.read_csv(result_path.parent / f"stream-{data_name}_dataset.csv")
        mask_prop = dataset_info["mask"].sum() / dataset_info.shape[0]
    except:
        mask_prop = -1
    try:
        results_timestamps = pd.read_pickle(result_path.parent / f"time_index.pickle")
    except:
        results_timestamps = None
    res = pd.read_csv(result_path, header=None)

    if results_timestamps is not None:
        # # print(len(list(results_timestamps.values)))
        # timestamps = list(results_timestamps.values)[-res.shape[0]:]
        print(res.shape)
        # # print(len(timestamps))
        # # res.index = pd.to_datetime(timestamps)
        # aux_info['date_time'] = pd.to_datetime(aux_info['date_time'])
        print(res.head())
        res.set_index(0)
        res = res.loc[~res.index.duplicated(keep='first')]
        print(res.head())
        timestamps = results_timestamps.iloc[-res.shape[0]:]
        print(timestamps.shape)
        # print(aux_info['date_time'])
        res.index = timestamps
        res.index = pd.to_datetime(timestamps, utc =False).dt.tz_localize(None)
        aux_info['date_time'] = pd.to_datetime(aux_info['date_time'], utc=False).dt.tz_localize(None)
        # print(res.index)
        # print(aux_info['date_time'])
        full_results = pd.merge(res, aux_info, how = 'left', left_index=True, right_on='date_time')
    else:
        full_results = pd.merge(res, aux_info, how = 'left', left_index=True, right_index = True)
    # print(full_results.head())

    # test_feature = 'PRES'
    res = {}
    # tfs = ['WD_1', 'WS_1', 'WD_3', 'WS_3', 'WD_4', 'WS_4']
    # if feature_set == "BMS":
    #     tfs = [x for x in list(full_results.columns) if 'WSPM' in str(x)]
    #     fs = ['h', 'day','wd', 'we']
    # if feature_set == "synth":
    #     fs = [('ground_truth_concept', 'Ground Truth')]
    tfs = []
    # for test_feature in ['WD_1', 'WS_1', 'WD_3', 'WS_3', 'WD_4', 'WS_4']:
    # for test_feature in [x for x in list(full_results.columns) if 'WSPM' in str(x)]:
    for test_feature in tfs:
        # test_feature = 'WD_1'
        # print(full_results.columns)
        full_results[test_feature] = pd.qcut(full_results[test_feature], 12, labels=False, duplicates="drop")
        # print(full_results.head(5))
        # print(run_options)
        res = {}
        for v in full_results[test_feature].unique():
            # print(f"{test_feature}, {v}")
            subset = full_results.loc[full_results[test_feature] == v]
            # if 'tree' not in str(result_path):
            #     prediction = subset[3][30000:]
            #     label = subset[4][30000:]
            # else:
            prediction = subset[3]
            label = subset[4]


            cm = get_confusion_matrix(prediction, label, levels)
            # print(cm)
            row_sums = []
            for r in cm:
                row_sums.append(np.sum(r))
            # print(row_sums)
            col_sums = []
            for c in cm.transpose():
                col_sums.append(np.sum(c))
            # print(col_sums)
            
            total = sum(row_sums)
            # print(total)
            chance_same_sum = 0
            for i,r in enumerate(row_sums):
                col_chance = col_sums[i] / total
                # print(col_chance)
                chance_same_sum += col_chance * r
                # print(chance_same_sum)
            pc = chance_same_sum / total
            # print(pc)

            pa_sum = sum([r[i] for i,r in enumerate(cm)])
            pa = pa_sum / total

            kappa = (pa - pc) / (1 - pc)
            # print(kappa)
            # res = {**res, "test_feature": test_feature, f"test_level": v, "pa": pa, f"pc": pc, f"kappa": kappa}
            res = {"test_feature": test_feature, f"test_level": v, "pa": pa, f"pc": pc, f"kappa": kappa}

            row = {**run_options, "dataset_name": rwf, "dataset_target_index": ti, 'fseed': seed, 'mask_p': mask_prop, **res, **results_from_json}

            rows.append(row)
            
    subset = full_results
    # prediction = subset[3][-30000:]
    # label = subset[4][-30000:]
    # prediction = subset[3]
    # label = subset[4]
    if 'tree' not in run_options['ct']:
        prediction = subset[3][20000:]
        label = subset[4][20000:]
    else:
        prediction = subset[3]
        label = subset[4]
    cm = get_confusion_matrix(prediction, label, levels)
    # print(cm)
    
    row_sums = []
    for r in cm:
        row_sums.append(np.sum(r))
    # print(row_sums)
    col_sums = []
    for c in cm.transpose():
        col_sums.append(np.sum(c))
    # print(col_sums)
    
    total = sum(row_sums)
    # print(total)
    chance_same_sum = 0
    for i,r in enumerate(row_sums):
        col_chance = col_sums[i] / total
        # print(col_chance)
        chance_same_sum += col_chance * r
        # print(chance_same_sum)
    pc = chance_same_sum / total
    # print(pc)

    pa_sum = sum([r[i] for i,r in enumerate(cm)])
    pa = pa_sum / total
    # print(pa)
    # print(full_results.tail(2))
    # exit()
    kappa = (pa - pc) / (1 - pc)
    # if np.isnan(kappa):
        # print(result_path)
        # print(full_results.head())
        # print(kappa)
    # res = {**res, "test_feature": "all", f"test_level": 0, "pa": pa, f"pc": pc, f"kappa": kappa}
    res = {"test_feature": "all", f"test_level": 0, "pa": pa, f"pc": pc, f"kappa": kappa}

    row = {**run_options, "dataset_name": rwf, "dataset_target_index": ti, 'fseed':seed, 'mask_p': mask_prop, **res, **results_from_json}
    rows.append(row)
    # print(row)
df = pd.DataFrame(rows)
df['feature_set'] = df['ct'].values
df['feature_set'] = df['feature_set'] + df['backtrack'].astype('str').values
df['feature_set'] = df['feature_set'] + df['proactive_sensitivity'].astype('str').values
df['feature_set'].loc[df['feature_set'].str.contains('linear')] = 'linear'
df['feature_set'].loc[df['feature_set'].str.contains('temporal')] = 'temporal'

#%%
# df.groupby(['feature_set', 'window']).mean()[['pa', 'kappa']]
#%%
# print(df.groupby('feature_set').mean().head()['pa', 'kappa'])
table = None
print(df.groupby('feature_set').mean().head())
kappa_df = df.set_index(list(index_names))
print(kappa_df.groupby('ct').mean().head())
# kappa_df = kappa_df.reset_index(['dataset_name'])
# kappa_df = kappa_df.drop('dataset_name', axis = 1)
kappa_df.head(10)

#%%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

by_ct = kappa_df.groupby(['feature_set', 'dataset_name','window']).mean()
# by_ct = kappa_df.groupby(['feature_set', 'dataset_name', 'css', 'csd', 'atp', 'window']).agg(['mean', 'std'])
# by_ct = kappa_df.groupby(['dataset_name', 'dataset_target_index', 'fseed', 'feature_set' , 'css', 'csd', 'atp', 'window']).agg(['mean', 'std'])
by_ct = kappa_df.groupby(['dataset_name', 'feature_set' , 'css', 'csd', 'atp', 'window']).agg(['mean', 'std'])
by_ct = kappa_df.groupby(['dataset_name', 'dataset_target_index', 'feature_set' , 'css', 'csd', 'atp', 'window']).agg(['mean', 'std'])
# print(kappa_df.groupby(['feature_set', 'dataset_name', 'css', 'csd', 'atp', 'window']).size())
print(kappa_df.groupby(['feature_set', 'dataset_name', 'css', 'csd', 'atp', 'window']).mean()[['pa', 'kappa']])
print(kappa_df.groupby(['feature_set', 'dataset_name', 'mask_p', 'css', 'csd', 'atp', 'window']).mean()[['pa', 'kappa']])
print(kappa_df.groupby(['feature_set', 'dataset_name', 'css', 'csd', 'atp', 'window']).size())
table_add = by_ct.unstack('dataset_name')[['pa', 'kappa']].swaplevel(0, 1, axis = 1)
print(table_add)
by_ct = kappa_df.groupby(['dataset_name', 'feature_set' , 'css', 'csd', 'atp', 'window']).agg(['mean', 'std'])
table_add = by_ct.unstack('dataset_name')[['pa', 'kappa']].swaplevel(0, 1, axis = 1)
print(table_add)

#%%
# kappa_df = kappa_df.xs(0.15, level = 'css', drop_level=False)
# kappa_df = kappa_df.iloc[kappa_df.index.get_level_values('css').isin([0.15, -1])]
# print(kappa_df.head())
# kappa_df = kappa_df.loc[kappa_df['test_feature'] == 'all']
# # print(kappa_df.head())
# # print(results.head(10))
# # results = results.reset_index('dataset_name')
# merged = results.join(kappa_df)
# print(merged.head())
# print(merged.head())
# by_ct = merged.groupby(['feature_set', 'dataset_name','window']).mean()
# by_ct = merged.groupby(['feature_set', 'dataset_name', 'css', 'csd', 'atp', 'window']).agg(['mean', 'std'])
# print(merged.groupby(['feature_set', 'dataset_name', 'css', 'csd', 'atp', 'window']).size())
# table_add = by_ct.unstack('dataset_name')[['pa', 'kappa']].swaplevel(0, 1, axis = 1)
# # print(table_add.head(10))
# if table is None:
#     table = table_add
# else:
#     table = table.join(table_add)

# print(table)

table_add.to_csv(pathlib.Path.cwd() / f"{data_name}-performance-table_sub.csv")

# %%

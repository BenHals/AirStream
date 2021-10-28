#%%
import pathlib
import json
import argparse
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
def get_concept_accuracy_from_df(data):
    system_concept_accuracy_sum = 0
    concept_compositions = {}
    concept_purity = []
    do_purity = False
    if do_purity:
        for i, row in data.iterrows():
            sample_gt_concept = row['ground_truth_concept']
            sample_sys_concept = row['system_concept']
            reuse = sample_sys_concept in concept_compositions
            concept_com = concept_compositions.setdefault(sample_sys_concept, Counter())
            concept_com[sample_gt_concept] += 1
            purity_against_gt = concept_com[sample_gt_concept] / sum(concept_com.values())
            if reuse:
                system_concept_accuracy_sum += purity_against_gt
            concept_purity.append(purity_against_gt)
    else:
        system_concept_accuracy_sum = 0
        concept_purity = np.ones(data.shape[0])
    system_concept_accuracy = system_concept_accuracy_sum / data.shape[0]
    return system_concept_accuracy, np.array(concept_purity)

def get_concept_transparency_data(ground_truth, system, purity, merge_key):
    # print(ground_truth[:5])
    # print(system[:5])
    # print(purity[:5])

    gt_values, gt_total_counts = np.unique(ground_truth, return_counts = True)
    print(system)
    sys_values, sys_total_counts = np.unique(system, return_counts = True)

    # print(gt_values)
    # print(sys_values)
    matrix = np.array([ground_truth, system, purity]).transpose()
    # print(matrix[:5])
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
        # print(f"gt = {gt}")
        gt_total_count = gt_total_counts[gt_i]
        # print(f'gt_total_count {gt_total_count}')
        gt_mask = matrix[matrix[:,0] == gt]
        sys_by_gt_values, sys_by_gt_counts = np.unique(gt_mask[:, 1], return_counts = True)
        # print(f"GT SHOULD BE THE SAME {gt_total_count}:{gt_mask.shape[0]}")
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
            # print(f'sys_total_count {sys_total_count}')
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
        # print(" GT recalls")
        # print(recall_values)
        # print("GT precisions")
        # print(precision_values)
        gt_result = {
            'Max Recall': max_recall,
            'Max Precision': max_precision,
            'Precision for Max Recall': precision_max_recall,
            'Recall for Max Precision': recall_max_precision,
            'f1': max_f1
        }
        # print(" GT result")
        # print(gt_result)
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
        # print("Sys result")
        # print(sys_result)
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
data_name = "BMSClean"
target_index = -1
target_seed = -1
experiment_directory = pathlib.Path.cwd() / 'experiments' / data_name
if target_index >= 0:
    experiment_directory = experiment_directory / str(target_index)
if target_seed >= 0:
    experiment_directory = experiment_directory / str(target_seed)
print(experiment_directory)
data_name = "BMSClean"

#%%
data_name = "WINDSIM"
experiment_directory = pathlib.Path("H:/PhDTest/NECTARDATA/synthetic") / data_name
# data_name = "RangioraClean"
# data_name = "BMSClean"
data_name = "Arrowtown"
# experiment_directory = pathlib.Path("../RepairResearch/ICDM/ReturnScaleFin/experiments") / data_name
experiment_directory = pathlib.Path(r"H:\PhD\Paper2-AdaptionRepair\ReturnScaleFin\experiments") / data_name
experiment_directory = experiment_directory.resolve()
target_index = -1
target_seed = -1
ut = False
print(experiment_directory)
#%%
parser = argparse.ArgumentParser()
parser.add_argument("-o", default="cwd", type=str, help="The directory containing the experiment")
parser.add_argument("-rd", default="RangioraClean", type=str, help="The directory containing the experiment")
parser.add_argument("-rdr", default="experiments", type=str, help="The directory containing the experiment")
parser.add_argument("-dn", default="RangioraClean", type=str, help="The directory containing the experiment")
parser.add_argument("-ti", default=-1, type=int, help="The directory containing the experiment")
parser.add_argument("-seed", default=-1, type=int, help="The directory containing the experiment")
parser.add_argument("-fs", default="rang", type=str, help="Feature set")
parser.add_argument("-ut", action="store_true", help="Feature set")

args = parser.parse_args()

# data_name = "Rangiora"
data_name = args.rd
target_index = args.ti
target_seed = args.seed
feature_set = args.fs
ut = args.ut
# data_name = "BMS"
origin_path = pathlib.Path.cwd() if args.o == 'cwd' else pathlib.Path(args.o).resolve()

experiment_directory = origin_path / args.rdr / data_name
if target_index >= 0:
    experiment_directory = experiment_directory / str(target_index)
if target_seed >= 0:
    experiment_directory = experiment_directory / str(target_seed)
print(experiment_directory)
data_name = args.dn

#%%
all_files = experiment_directory.glob('**/*-results.json')
if ut:
    data_files = experiment_directory.glob('**/dataset_target_results.json')
    res = {}
    for data_f in data_files:
        res = {**json.load(data_f.open()), ** res}
    # print(res)
    # print(len(list(res.keys())))
    all_results = res
    print(all_results)
    formatted_results = {}
    index_names = []
    for k in all_results:
        key = json.loads(k)
        if 'gitcommit' in k:
            del key['gitcommit']
        if 'dataset_target_index' in k:
            del key['dataset_target_index']
        index_names = key.keys()
        print(len(list(index_names)))
        print(key)
        print(list(index_names))

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
else:
    all_files = list(experiment_directory.glob("**/*-results.json"))
    formatted_results = {}
    for data_f in all_files:
        print(data_f)
        try:
            k = json.load(data_f.open())
        except:
            continue
        try:
            options_str = f"{str(data_f)[:-13]}_run_options.txt"
            key = json.load(pathlib.Path(options_str).open())
        except Exception as e:
            continue
            print(e)
            print(str(data_f).split('.')[0])
            # options_str = f"{str(data_f).split('.')[0]}_run_options.txt"
            options_str = data_f.parent / f"{str(data_f.stem).split('-results')[0]}_run_options.txt"
            key = json.load(pathlib.Path(options_str).open())
        index_names = list(key.keys())[:19]
        print(k)
        print(key.values())
        formatted_results[tuple(list(key.values())[:19])] = {**k}
        pointer = formatted_results[tuple(list(key.values())[:19])]
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

#%%
# level_counts = []
# for result_path in results_files:
#     if data_name in str(result_path.stem) or 'full_link' in str(result_path.stem) or 'drift_info' in str(result_path.stem) or 'stream' in str(result_path.stem):
#         continue
#     res = pd.read_csv(result_path, header=None)
#     # print(res.head())
#     # print(res.head(3))
#     level_counts.append(res[3].nunique())
# # print(level_counts)
# levels = set(list(range(max(level_counts))))
failed = set()
# feature_set = 'rang'
# feature_set = 'BMS'
feature_set = 'Arrow'
levels = [0, 1, 2, 3, 4, 5]
if feature_set == "synth":
    levels = [0, 1]
    if data_name == "WINDSIM":
        levels = [0, 1, 2, 3, 4]
levels = set(levels)
#%%
from tqdm import tqdm, tqdm_notebook
import sklearn
import sklearn.ensemble
import sklearn.svm
import sklearn.naive_bayes
import sklearn.model_selection
import sklearn.metrics
import sklearn.dummy
rows = []
results_files = list(experiment_directory.glob('**/*.csv'))
top_result_path_f1_score = None
top_result_path = None
# for result_path in picked_results_files:
for result_path in tqdm(results_files):
    # print(result_path)
    if data_name in str(result_path.stem) or 'full_link' in str(result_path.stem) or 'drift_info' in str(result_path.stem) or 'stream' in str(result_path.stem) or 'aux' in str(result_path.stem):
        continue
    
    if 'ds1_bt' in str(result_path) and (not ('ds1_bt_amend' in str(result_path))):
        continue

    # if (not ('ds1' in str(result_path))) and (not 'sys' in str(result_path)):
    #     continue
    if (not ('dynse' in str(result_path) or 'sys' in str(result_path))):
        continue

    rwf = result_path.parent.parent.parent.stem
    seed = int(result_path.parent.stem)
    try:
        ti = int(result_path.parent.parent.stem)
    except:
        ti = 0
        rwf = result_path.parent.parent.stem

    # df_info_path = result_path.parent / f"df_info.json"
    # df_info = json.load(df_info_path.open())
    # print(df_info)
    # if df_info['target_sensor'] != "102":
    #     continue
    run_options = result_path.parent / f"{result_path.stem}_run_options.txt"
    try:
        run_options = json.load(run_options.open())
    except:
        print(result_path)
        continue
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
    except:
        with open(result_path.parent / f"stream-{seed}.csv") as f:
            data = f.read()
            dataset_info = pd.read_csv(io.StringIO(data.replace(',\n','\n')), header = None)
        # dataset_info = pd.read_csv(result_path.parent / f"stream-{seed}.csv", header = None)
    try:
        results_timestamps = pd.read_pickle(result_path.parent / f"time_index.pickle")
        print(f"{result_path} - {'dynse' in str(result_path)}")
        if 'dynse' not in str(result_path):
            res = pd.read_csv(result_path, header=None)
        else:
            print("using header")
            res = pd.read_csv(result_path, header=0)
        print(res.head())

        # print(len(list(results_timestamps.values)))
        timestamps = list(results_timestamps.values)[-res.shape[0]:]
        # print(res.shape)
        # print(len(timestamps))
        res.index = pd.to_datetime(timestamps)
        aux_info['date_time'] = pd.to_datetime(aux_info['date_time'])
        full_results = pd.merge(res, aux_info, how = 'left', left_index=True, right_on='date_time')
    except:
        if 'dynse' not in str(result_path):
            res = pd.read_csv(result_path, header=None)
        else:
            print("using header")
            res = pd.read_csv(result_path, header=0)
        full_results = pd.merge(res, aux_info, how = 'left', left_index=True, right_index=True)

    # test_feature = 'PRES'
    data = {}

    
# h,day,wd,we
    fs = [('WD_1', 'Wind Direction'), ('WS_1', 'Wind Speed'), ('WD_3', 'Wind Direction'), ('WS_3', 'Wind Speed'), ('WD_4', 'Wind Direction'), ('WS_4', 'Wind Direction')]
    if feature_set == "BMS":
        fs = [('wd', 'Wind Direction'), ('WSPM', 'Wind Speed'), ("TEMP", "Temperature"), ("PRES", "Pressure"), ('DEWP', 'Dew Point')]
    if feature_set == "Arrow":
        fs = [('h', 'Hour'), ('day', 'Daytime'), ('wd', 'Week Day'), ('we', 'Weekend')]
    if feature_set == "synth":
        fs = [('ground_truth_concept', 'Ground Truth')]
    # for test_feature,feature_type in [('WD_1', 'Wind Direction'), ('WS_1', 'Wind Speed'), ('WD_3', 'Wind Direction'), ('WS_3', 'Wind Speed'), ('WD_4', 'Wind Direction'), ('WS_4', 'Wind Direction')]:
    # for test_feature,feature_type in [('h', 'Hour'), ('day', 'Daytime'), ('wd', 'Week Day'), ('we', 'Weekend')]:
    # for test_feature,feature_type in [('wd', 'Wind Direction'), ('WSPM', 'Wind Speed'), ("TEMP", "Temperature"), ("PRES", "Pressure"), ('DEWP', 'Dew Point')]:
    # for test_feature,feature_type in [('ground_truth_concept', 'Ground Truth')]:
    for test_feature,feature_type in fs:
        # test_feature = 'WD_1'
        # print(full_results.columns)
        if len(np.unique(full_results[test_feature])) > 16:
            full_results[test_feature] = pd.qcut(full_results[test_feature], 4, labels=False, duplicates="drop").values
        else:
            full_results[test_feature] = full_results[test_feature].astype("str")
        # full_results[test_feature] = pd.qcut(full_results[test_feature], 8, labels=False, duplicates="drop")
        # print(full_results.head(2))
        
        prediction_target = full_results[test_feature].to_numpy()
        features = dataset_info.iloc[-res.shape[0]:, :-1].to_numpy()
        if dataset_info.isnull().values.any():
            features = dataset_info.iloc[-res.shape[0]:, :-2].to_numpy()

        # print("features")
        # print(features[:10])
        # print(features.shape)
        # x = dataset_info[list(dataset_info.drop(list(dataset_info.columns)[-1], axis = 1).columns)].to_numpy()
        # print(x == features)
        # break
        # print(res.head(2))
        # print(dataset_info.head(2))
        # print(res.shape)
        # print(dataset_info.shape)
        if 'dynse' not in str(result_path):
            states = res[6].to_numpy()
        else:
            print(result_path)
            print(res.head())
            states = res['system_concept'].to_numpy()
        # print(prediction_target)
        # print(states)
        # print(features.shape)
        # print(states.shape)
        state_and_features = np.c_[features, states]
        # print(state_and_features.shape)
        
        feature_model = sklearn.ensemble.RandomForestClassifier(n_estimators=3,n_jobs=-1,max_depth=15)
        state_model = sklearn.ensemble.RandomForestClassifier(n_estimators=3)
        statefeature_model = sklearn.ensemble.RandomForestClassifier(n_estimators=3,n_jobs=-1,max_depth=15)
        dummy = sklearn.dummy.DummyClassifier(strategy = "prior")

        average_reuse_purity, concept_purity = get_concept_accuracy_from_df(pd.DataFrame.from_dict({'system_concept': states, 'ground_truth_concept': prediction_target}))
        # print(average_reuse_purity, concept_purity)
        gt_results, overall_results_a = get_concept_transparency_data(prediction_target , states , concept_purity, None)
        # print(gt_results)
        # print(overall_results_a)

        ds = []
        fs = []
        ss = []
        fss = []
        for i in range(1):
            # dummy_score = sklearn.model_selection.cross_val_score(dummy, prediction_target, prediction_target, scoring='accuracy', cv = 3)
            # feature_score = sklearn.model_selection.cross_val_score(feature_model, features, prediction_target, scoring='accuracy', cv = 3)
            # state_score = sklearn.model_selection.cross_val_score(state_model, states.reshape(-1, 1), prediction_target, scoring='accuracy', cv = 3)
            # statefeature_score = sklearn.model_selection.cross_val_score(statefeature_model, state_and_features, prediction_target, scoring='accuracy', cv = 3)
            # print(feature_score)
            # print(statefeature_score)
            # ds.append(np.mean(dummy_score))
            # fs.append(np.mean(feature_score))
            # ss.append(np.mean(state_score))
            # fss.append(np.mean(statefeature_score))

            fs.append(0)
            fss.append(0)

        # ds = full_results[f"link-{test_feature}-dummy_score"]
        # ds = full_results[f"link-{test_feature}-dummy_score"]
        # print(fs)
        # print(fss)
        # break
        for k in overall_results_a:
            data[f"{test_feature}-{k}"] = overall_results_a[k]
        data[f"{test_feature}-state_feature"] = np.mean(fss)
        data[f"{test_feature}-fs_test"] = np.mean(fs)
        # data = {**data, "test_feature": test_feature, "dummy": np.mean(ds), "feature": np.mean(fs), "state": np.mean(ss), "state_feature": np.mean(fss)}

    row = {**run_options, "dataset_name": rwf, "dataset_target_index": ti, **data}
    if 'backtrack' not in row:
        row['backtrack'] = False
    if 'proactive_sensitivity' not in row:
        row['proactive_sensitivity'] = False
    if 'window' not in row:
        row['window'] = -1
    if 'atp' not in row:
        row['atp'] = -1
    if 'css' not in row:
        row['css'] = -1
    if 'csd' not in row:
        row['csd'] = -1
    for k in results_from_json:
        if not isinstance(results_from_json[k],dict):
            row[k] = results_from_json[k]
        else:
            for rk in results_from_json[k]:
                row[f"{k}-{rk}"] = results_from_json[k][rk]
    # print(row)
    # if top_result_path_f1_score is None or top_result_path_f1_score < row['ground_truth_concept-f1']:
    #     top_result_path_f1_score = row['ground_truth_concept-f1']
    # if top_result_path_f1_score is None or top_result_path_f1_score < row['wd-f1'] + row['WSPM-f1']:
    #     top_result_path_f1_score = row['wd-f1'] + row['WSPM-f1']
    # if top_result_path_f1_score is None or top_result_path_f1_score < row['WD_4-f1'] + row['WS_4-f1']:
    #     top_result_path_f1_score = row['WD_4-f1'] + row['WS_4-f1']
    #     top_result_path = result_path
    rows.append(row)
df = pd.DataFrame(rows)
df.to_pickle(pathlib.Path.cwd() / f"{data_name}-condition-data_R2.pickle")

#%%
df = pd.read_pickle(pathlib.Path.cwd() / f"{data_name}-condition-data_R2.pickle")

df['feature_set'] = df['ct'].values
df['feature_set'] = df['feature_set'] + df['backtrack'].astype('str').values
df['feature_set'] = df['feature_set'] + df['proactive_sensitivity'].astype('str').values
df['feature_set'].loc[df['feature_set'].str.contains('linear')] = 'linear'
df['feature_set'].loc[df['feature_set'].str.contains('temporal')] = 'temporal'

#%%
df.head()
print(list(df.columns))
#%%
# kappa_df = df.set_index(list(index_names)).drop('feature_set', axis = 1)
kappa_df = df

#%%
# options_path = pathlib.Path('G:\My Drive\UniMine\Uni\PhD\RepairResearch\ICDM\ReturnScaleFin\experiments\RangioraClean\\10\\1473') / "ds1_bt_amend-cl_-1-an_5-w_500-sdp_500-msm_1#5-atl_1-atp_2000-bs_0#05-csd_0#2-css_0#2-True-False-0_run_options.txt"
options_path = pathlib.Path(r'G:\My Drive\UniMine\Uni\PhD\RepairResearch\ICDM\ReturnScaleFin\experiments\RangioraClean\10\1473') / "ds1_bt_amend-cl_-1-an_5-w_500-sdp_500-msm_1#5-atl_1-atp_2000-bs_0#05-csd_0#2-css_0#2-True-False-0_run_options.txt"
with options_path.open('r') as f:
    options = json.load(f)
    index_names = list(options.keys())
kappa_index = df.set_index(list(index_names)).drop('feature_set', axis = 1)

merged = results.join(kappa_index, rsuffix='_right')
merged.head()
#%%
merged = kappa_df
# print(merged['ground_truth_concept-f1'])
#%%
# merged = kappa_df
# print(list(merged.columns))
#%%
merged.groupby('feature_set').mean().head()
#%%
fs = [('WD_1', 'Wind Direction'), ('WS_1', 'Wind Speed'), ('WD_3', 'Wind Direction'), ('WS_3', 'Wind Speed'), ('WD_4', 'Wind Direction'), ('WS_4', 'Wind Direction')]
if feature_set == "BMS":
    fs = [('wd', 'Wind Direction'), ('WSPM', 'Wind Speed'), ("TEMP", "Temperature"), ("PRES", "Pressure"), ('DEWP', 'Dew Point')]
if feature_set == "Arrow":
    fs = [('h', 'Hour'), ('day', 'Daytime'), ('wd', 'Week Day'), ('we', 'Weekend')]
if feature_set == "synth":
    fs = [('ground_truth_concept', 'Ground Truth')]
# for test_feature,feature_type in [('WD_1', 'Wind Direction'), ('WS_1', 'Wind Speed'), ('WD_3', 'Wind Direction'), ('WS_3', 'Wind Speed'), ('WD_4', 'Wind Direction'), ('WS_4', 'Wind Direction')]:
# for test_feature,feature_type in [('h', 'Hour'), ('day', 'Daytime'), ('wd', 'Week Day'), ('we', 'Weekend')]:
# for test_feature,feature_type in [('wd', 'Wind Direction'), ('WSPM', 'Wind Speed'), ("TEMP", "Temperature"), ("PRES", "Pressure")]:
# for test_feature,feature_type in [('ground_truth_concept', 'Ground Truth')]:
for test_feature,feature_type in fs:
    # show_cols = [x for x in set(df.columns) if test_feature in x and 'f1' in x]
    if feature_type not in df.columns:
        merged[f"{feature_type}-f1"] = merged[f"{test_feature}-f1"]
    else:
        merged[f"{feature_type}-f1"] = merged[f"{feature_type}-f1"] + merged[f"{test_feature}-f1"]

    if feature_type not in merged.columns:
        merged[f"{feature_type}-model_score"] = merged[f"link-{test_feature}-model_score"]
    else:
        merged[f"{feature_type}-model_score"] = merged[f"{feature_type}-model_score"] + merged[f"link-{test_feature}-model_score"]
    if feature_type not in merged.columns:
        merged[f"{feature_type}-feature_increase"] = merged[f"link-{test_feature}-feature_increase"]
    else:
        merged[f"{feature_type}-feature_increase"] = merged[f"{feature_type}-feature_increase"] + merged[f"link-{test_feature}-feature_increase"]
    if feature_type not in merged.columns:
        merged[f"{feature_type}-dummy_score"] = merged[f"link-{test_feature}-dummy_score"]
    else:
        merged[f"{feature_type}-dummy_score"] = merged[f"{feature_type}-dummy_score"] + merged[f"link-{test_feature}-dummy_score"]
    if feature_type not in merged.columns:
        merged[f"{feature_type}-feature_score"] = merged[f"link-{test_feature}-model_score"] - merged[f"link-{test_feature}-feature_increase"]
    else:
        merged[f"{feature_type}-feature_score"] = merged[f"{feature_type}-feature_score"] + merged[f"link-{test_feature}-model_score"] - merged[f"link-{test_feature}-feature_increase"]
    
    if feature_type not in merged.columns:
        merged[f"{feature_type}-state_feature"] = merged[f"{test_feature}-state_feature"]
    else:
        merged[f"{feature_type}-state_feature"] = merged[f"{feature_type}-state_feature"] + merged[f"{test_feature}-state_feature"]
    
    if feature_type not in merged.columns:
        merged[f"{feature_type}-fs_test"] = merged[f"{test_feature}-fs_test"]
    else:
        merged[f"{feature_type}-fs_test"] = merged[f"{feature_type}-fs_test"] + merged[f"{test_feature}-fs_test"]
for test_feature,feature_type in fs:
    merged[f"{feature_type}-feature_increaseO"] = merged[f"{feature_type}-model_score"] - merged[f"{feature_type}-feature_score"]
    # fs = [('WD_1', 'Wind Direction'), ('WS_1', 'Wind Speed'), ('WD_3', 'Wind Direction'), ('WS_3', 'Wind Speed'), ('WD_4', 'Wind Direction'), ('WS_4', 'Wind Direction')]
show_cols = [f"Wind Direction-f1", "Wind Speed-f1",
             "Wind Direction-model_score", "Wind Direction-dummy_score", "Wind Direction-feature_score", "Wind Direction-state_feature", "Wind Direction-fs_test", "Wind Direction-feature_increase",
              "Wind Speed-model_score", "Wind Speed-dummy_score", "Wind Speed-feature_score", "Wind Speed-state_feature", "Wind Speed-fs_test", "Wind Speed-feature_increase"]
if feature_set == "BMS":
    show_cols = [f"Wind Direction-f1", "Wind Speed-f1", "Temperature-f1", "Pressure-f1",
    "Wind Direction-model_score", "Wind Direction-dummy_score", "Wind Direction-feature_score", "Wind Direction-feature_increase",
    "Wind Speed-model_score", "Wind Speed-dummy_score", "Wind Speed-feature_score", "Wind Speed-feature_increase",
    "Temperature-model_score", "Temperature-dummy_score", "Temperature-feature_score", "Temperature-feature_increase",
    "Pressure-model_score", "Pressure-dummy_score", "Pressure-feature_score", "Pressure-feature_increase",]
if feature_set == "Arrow":
    show_cols = ["Hour-f1", 'Hour-model_score', 'Hour-feature_score', "Hour-feature_increase",
            "Daytime-f1", 'Daytime-model_score', 'Daytime-feature_score', "Daytime-feature_increase",
            "Week Day-f1", 'Week Day-model_score', 'Week Day-feature_score', "Week Day-feature_increase",
            "Weekend-f1", 'Weekend-model_score', 'Weekend-feature_score', "Weekend-feature_increase"]
if feature_set == "synth":
    show_cols = [f"Ground Truth-f1", f"Ground Truth-model_score", f"Ground Truth-state_feature", f"Ground Truth-feature_score", f"Ground Truth-fs_test", "Ground Truth-feature_increase"]
# show_cols = [f"Wind Direction-f1", "Wind Speed-f1",
            #  "Wind Direction-model_score", "Wind Direction-dummy_score", "Wind Direction-feature_score", "Wind Direction-state_feature", "Wind Direction-fs_test",
            #   "Wind Speed-model_score", "Wind Speed-dummy_score", "Wind Speed-feature_score", "Wind Speed-state_feature", "Wind Speed-fs_test"]
# show_cols = [f"Wind Direction-f1", "Wind Speed-f1", "Temperature-f1", "Pressure-f1", "Wind Direction-model_score", "Wind Direction-dummy_score", "Wind Direction-feature_score", "Wind Speed-model_score", "Wind Speed-dummy_score", "Wind Speed-feature_score", "Temperature-model_score", "Temperature-dummy_score", "Temperature-feature_score", "Pressure-model_score", "Pressure-dummy_score", "Pressure-feature_score"]
# show_cols = [f"Ground Truth-f1", f"Ground Truth-model_score", f"Ground Truth-state_feature", f"Ground Truth-feature_score", f"Ground Truth-fs_test"]
# [('h', 'Hour'), ('day', 'Daytime'), ('wd', 'Week Day'), ('we', 'Weekend')]
# show_cols = ["Hour-f1", 'Hour-model_score', 'Hour-feature_score',
#             "Daytime-f1", 'Daytime-model_score', 'Daytime-feature_score',
#             "Week Day-f1", 'Week Day-model_score', 'Week Day-feature_score',
#             "Weekend-f1", 'Weekend-model_score', 'Weekend-feature_score']
table = merged.groupby(['feature_set', 'window', 'atp', 'css', 'csd']).aggregate(['mean', 'std'])[show_cols]
table
#%%
# table = None
# kappa_df = df.set_index(list(index_names)).drop('feature_set', axis = 1)
# # kappa_df = kappa_df.xs(0.15, level = 'css', drop_level=False)
# # kappa_df = kappa_df.iloc[kappa_df.index.get_level_values('css').isin([0.15, -1])]
# # print(kappa_df.head())
# kappa_df = kappa_df.loc[kappa_df['test_feature'] == 'all']
# # print(kappa_df.head())
# # print(results.head(10))
# merged = results.join(kappa_df)
# # by_ct = merged.groupby(['feature_set', 'dataset_name', 'css', 'csd', 'window']).mean()
# by_ct = merged.groupby(['feature_set', 'dataset_name', 'css', 'csd', 'window']).agg(['mean', 'std'])
# print(merged.groupby(['feature_set', 'dataset_name', 'css', 'csd', 'window']).size())
# table_add = by_ct.unstack('dataset_name')[['pa', 'kappa']].swaplevel(0, 1, axis = 1)
# # print(table_add.head(10))
# if table is None:
#     table = table_add
# else:
#     table = table.join(table_add)

# print(table)
# kappa_df = df.set_index(list(index_names)).drop('feature_set', axis = 1)
# print(kappa_df.head())
# print(results.head(10))
# merged = results.join(kappa_df)
# print(list(merged.columns))
# print(merged.head(2))
# print(merged['WD_1-f1'])
# for test_feature,feature_type in [('WD_1', 'Wind Direction'), ('WS_1', 'Wind Speed'), ('WD_3', 'Wind Direction'), ('WS_3', 'Wind Speed'), ('WD_4', 'Wind Direction'), ('WS_4', 'Wind Direction')]:
#     # show_cols = [x for x in set(df.columns) if test_feature in x and 'f1' in x]
#     if feature_type not in df.columns:
#         merged[f"{feature_type}-f1"] = merged[f"{test_feature}-f1"]
#     else:
#         merged[f"{feature_type}-f1"] = merged[f"{feature_type}-f1"] + merged[f"{test_feature}-f1"]

#     if feature_type not in merged.columns:
#         merged[f"{feature_type}-model_score"] = merged[f"link-{test_feature}-model_score"]
#     else:
#         merged[f"{feature_type}-model_score"] = merged[f"{feature_type}-model_score"] + merged[f"link-{test_feature}-model_score"]
#     if feature_type not in merged.columns:
#         merged[f"{feature_type}-dummy_score"] = merged[f"link-{test_feature}-dummy_score"]
#     else:
#         merged[f"{feature_type}-dummy_score"] = merged[f"{feature_type}-dummy_score"] + merged[f"link-{test_feature}-dummy_score"]
#     if feature_type not in merged.columns:
#         merged[f"{feature_type}-feature_score"] = merged[f"link-{test_feature}-model_score"] - merged[f"link-{test_feature}-feature_increase"]
#     else:
#         merged[f"{feature_type}-feature_score"] = merged[f"{feature_type}-feature_score"] + merged[f"link-{test_feature}-model_score"] - merged[f"link-{test_feature}-feature_increase"]
# show_cols = [f"Wind Direction-f1", "Wind Speed-f1", "Wind Direction-model_score", "Wind Direction-dummy_score", "Wind Direction-feature_score", "Wind Speed-model_score", "Wind Speed-dummy_score", "Wind Speed-feature_score"]
# show_cols = [f"Wind Direction-f1", "Wind Speed-f1", "Wind Direction-model_score", "Wind Direction-dummy_score", "Wind Direction-feature_score", "Wind Speed-model_score", "Wind Speed-dummy_score", "Wind Speed-feature_score"]
# table = merged.groupby(['feature_set', 'window']).mean()[show_cols]
# table = merged.groupby(['feature_set', 'dataset_name', 'css', 'csd', 'window']).agg(['mean', 'std'])[show_cols]
# idx = table.columns.str.split('-', expand=True)
# table.columns = idx
# idx = pd.MultiIndex.from_product([idx.levels[0], idx.levels[1]])
# table.reindex(columns=idx, fill_value=-1)
# table.stack(0)
print(table)
# table.stack('feature_set')
# table.transpose()


#%%
table.to_csv(pathlib.Path.cwd() / f"{data_name}-condition-table_R2.csv")

# %%

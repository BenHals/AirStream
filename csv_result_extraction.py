import numpy as np
import pandas as pd
import os, time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from graphviz import Digraph
from collections import Counter
sns.set()



version = 1
def get_concept_accuracy_from_df(data, merge_key):
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

def get_drift_point_accuracy(data, follow_length = 250):
    print("Drift point locations")
    if not 'drift_occured' in data.columns:
        print("no column")
        return 0, 0, 0
    dpl = data.index[data['drift_occured'] == 1].tolist() 
    dpl = dpl[1:]  
    print(dpl)
    if len(dpl) < 1:
        return 0, 0, 0
    # change_detected
    # def close_to_point(val):
    #     for point in dpl:
    #         if val < point:
    #             return False
    #         if val < point + 1000:
    #             return True
    #     return False
    
    # filtered_index = data.index.map(close_to_point)
    # filtered = data.iloc[filtered_index]

    start_shift_time = time.time()
    following_drift = np.unique(np.concatenate([np.arange(i, min(i+follow_length+1, len(data))) for i in dpl]))
    
    filtered = data.iloc[following_drift]
    end_shift_time = time.time()
    print(f"NP index vals took {end_shift_time - start_shift_time}")

    num_close = filtered.shape[0]
    sum_correct = filtered['is_correct'].sum()
    close_accuracy = sum_correct / num_close
    print(filtered)
    values, counts = np.unique(filtered['y'], return_counts = True)
    print(values, counts)
    majority_class = values[np.argmax(counts)]
    print(majority_class)
    majority_correct = filtered.loc[filtered['y'] == majority_class]
    print(majority_correct.shape[0])
    num_majority_correct = majority_correct.shape[0]
    majority_accuracy =  num_majority_correct / num_close
    print(majority_accuracy)
    kappa_m = (close_accuracy - majority_accuracy) / (1 - majority_accuracy)
    print(f"Kappa_m = {kappa_m}")
    temporal_filtered = filtered['y'].shift(1, fill_value = 0.0)
    print(temporal_filtered.head(5))
    print(filtered['y'].head(5))
    filtered_correct = filtered['y'] == temporal_filtered
    # print(filtered_correct)
    temporal_accuracy = filtered_correct.sum() / num_close
    print(temporal_accuracy)
    kappa_t = (close_accuracy - temporal_accuracy) / (1 - temporal_accuracy)
    print(f"Kappa_t = {kappa_t}")

    # print()
    print(f"Num close to drift {num_close}")
    print(f"sum close to drift {sum_correct}")
    print(f"acc close to drift {close_accuracy}")
    return close_accuracy, kappa_m, kappa_t

def get_detect_point_accuracy(data, follow_length = 250):
    print("detect point locations")
    
    if not 'change_detected' in data.columns:
        print("no column")
        return 0, 0, 0
    dpl = data.index[data['change_detected'] == 1].tolist()   
    if len(dpl) < 1:
        print("No detections")
        return 0, 0, 0
    print(dpl)
    # change_detected

    # start_func_time = time.time()
    # def close_to_point(val):
    #     for point in dpl:
    #         if val < point:
    #             return False
    #         if val < point + follow_length:
    #             return True
    #     return False
    # filtered_index = data.index.map(close_to_point)
    # filtered = data.iloc[filtered_index]
    # # print(filtered)
    # end_func_time = time.time()
    # print(f"Filtering function took {end_func_time - start_func_time}")
    # num_close = filtered.shape[0]
    # sum_correct = filtered['is_correct'].sum()
    # close_accuracy = sum_correct / num_close
    # # print()
    # print(f"Num close to drift {num_close}")
    # print(f"sum close to drift {sum_correct}")
    # print(f"acc close to drift {close_accuracy}")


    # start_shift_time = time.time()
    # following_drift = data['change_detected']
    # for i in range(follow_length):
    #     following_drift = following_drift + data['change_detected'].shift(periods = i + 1, fill_value = 0)
    # # print(following_drift >= 1)
    
    # filtered = data.loc[following_drift >= 1]
    # end_shift_time = time.time()
    # print(f"Shifting took {end_shift_time - start_shift_time}")
    # # print(filtered)

    # num_close = filtered.shape[0]
    # sum_correct = filtered['is_correct'].sum()
    # close_accuracy = sum_correct / num_close
    # # print()
    # print(f"Num close to drift {num_close}")
    # print(f"sum close to drift {sum_correct}")
    # print(f"acc close to drift {close_accuracy}")


    start_shift_time = time.time()
    following_drift = np.unique(np.concatenate([np.arange(i, min(i+follow_length+1, len(data))) for i in dpl]))
    
    filtered = data.iloc[following_drift]
    end_shift_time = time.time()
    print(f"NP index vals took {end_shift_time - start_shift_time}")
    # print(filtered)

    num_close = filtered.shape[0]
    sum_correct = filtered['is_correct'].sum()
    close_accuracy = sum_correct / num_close
    print(filtered)
    values, counts = np.unique(filtered['y'], return_counts = True)
    print(values, counts)
    majority_class = values[np.argmax(counts)]
    print(majority_class)
    majority_correct = filtered.loc[filtered['y'] == majority_class]
    print(majority_correct.shape[0])
    num_majority_correct = majority_correct.shape[0]
    majority_accuracy =  num_majority_correct / num_close
    print(majority_accuracy)
    kappa_m = (close_accuracy - majority_accuracy) / (1 - majority_accuracy)
    print(f"Kappa_m = {kappa_m}")
    temporal_filtered = filtered['y'].shift(1, fill_value = 0.0)
    print(temporal_filtered.head(5))
    print(filtered['y'].head(5))
    filtered_correct = filtered['y'] == temporal_filtered
    # print(filtered_correct)
    temporal_accuracy = filtered_correct.sum() / num_close
    print(temporal_accuracy)
    kappa_t = (close_accuracy - temporal_accuracy) / (1 - temporal_accuracy)
    print(f"Kappa_t = {kappa_t}")

    # print()
    print(f"Num close to drift {num_close}")
    print(f"sum close to drift {sum_correct}")
    print(f"acc close to drift {close_accuracy}")
    return close_accuracy, kappa_m, kappa_t

def get_uniquedetect_point_accuracy(data, follow_length = 250):
    print("Udetect point locations")
    
    if not 'change_detected' in data.columns:
        print("no column")
        return 0, 0, 0
    dpl = data.index[data['change_detected'] == 1].tolist()
    dpl_spaced = []
    for i, point in enumerate(dpl[:-1]):
        if dpl[i + 1] > point + 500:
            dpl_spaced.append(point)
    dps = dpl_spaced
    if len(dpl) < 1:
        print("No detections")
        return 0, 0, 0
    print(dpl)
    start_shift_time = time.time()
    following_drift = np.unique(np.concatenate([np.arange(i, min(i+follow_length+1, len(data))) for i in dpl]))
    
    filtered = data.iloc[following_drift]

    
    
    end_shift_time = time.time()
    print(f"NP index vals took {end_shift_time - start_shift_time}")
    # print(filtered)

    num_close = filtered.shape[0]
    sum_correct = filtered['is_correct'].sum()
    close_accuracy = sum_correct / num_close
    print(filtered)
    values, counts = np.unique(filtered['y'], return_counts = True)
    print(values, counts)
    majority_class = values[np.argmax(counts)]
    print(majority_class)
    majority_correct = filtered.loc[filtered['y'] == majority_class]
    print(majority_correct.shape[0])
    num_majority_correct = majority_correct.shape[0]
    majority_accuracy =  num_majority_correct / num_close
    print(majority_accuracy)
    kappa_m = (close_accuracy - majority_accuracy) / (1 - majority_accuracy)
    print(f"Kappa_m = {kappa_m}")
    temporal_filtered = filtered['y'].shift(1, fill_value = 0.0)
    print(temporal_filtered.head(5))
    print(filtered['y'].head(5))
    filtered_correct = filtered['y'] == temporal_filtered
    # print(filtered_correct)
    temporal_accuracy = filtered_correct.sum() / num_close
    print(temporal_accuracy)
    kappa_t = (close_accuracy - temporal_accuracy) / (1 - temporal_accuracy)
    print(f"Kappa_t = {kappa_t}")

    # print()
    print(f"Num close to drift {num_close}")
    print(f"sum close to drift {sum_correct}")
    print(f"acc close to drift {close_accuracy}")
    return close_accuracy, kappa_m, kappa_t

def get_driftdetect_point_accuracy(data, follow_length = 250):
    print("Ddetect point locations")
    
    if not 'change_detected' in data.columns:
        print("no column")
        return 0, 0, 0
    if not 'drift_occured' in data.columns:
        print("no column")
        return 0, 0, 0
    dpl = data.index[data['change_detected'] == 1].tolist()   
    drift_indexes = data.index[data['drift_occured'] == 1].tolist()   
    if len(dpl) < 1:
        print("No detections")
        return 0, 0, 0
    print(dpl)
    start_shift_time = time.time()
    following_drift = np.unique(np.concatenate([np.arange(i, min(i+1000+1, len(data))) for i in drift_indexes]))
    following_detect= np.unique(np.concatenate([np.arange(i, min(i+follow_length+1, len(data))) for i in dpl]))


    following_both = np.intersect1d(following_detect, following_drift, assume_unique= True)

    print(following_both)
    
    filtered = data.iloc[following_both]
    end_shift_time = time.time()
    print(f"NP index vals took {end_shift_time - start_shift_time}")
    # print(filtered)

    num_close = filtered.shape[0]
    if num_close == 0:
        return 0, 0, 0
    sum_correct = filtered['is_correct'].sum()
    close_accuracy = sum_correct / num_close
    print(filtered)
    values, counts = np.unique(filtered['y'], return_counts = True)
    print(values, counts)
    majority_class = values[np.argmax(counts)]
    print(majority_class)
    majority_correct = filtered.loc[filtered['y'] == majority_class]
    print(majority_correct.shape[0])
    num_majority_correct = majority_correct.shape[0]
    majority_accuracy =  num_majority_correct / num_close
    print(majority_accuracy)
    kappa_m = (close_accuracy - majority_accuracy) / (1 - majority_accuracy)
    print(f"Kappa_m = {kappa_m}")
    temporal_filtered = filtered['y'].shift(1, fill_value = 0.0)
    print(temporal_filtered.head(5))
    print(filtered['y'].head(5))
    filtered_correct = filtered['y'] == temporal_filtered
    # print(filtered_correct)
    temporal_accuracy = filtered_correct.sum() / num_close
    print(temporal_accuracy)
    kappa_t = (close_accuracy - temporal_accuracy) / (1 - temporal_accuracy)
    print(f"Kappa_t = {kappa_t}")

    # print()
    print(f"Num close to drift {num_close}")
    print(f"sum close to drift {sum_correct}")
    print(f"acc close to drift {close_accuracy}")
    return close_accuracy, kappa_m, kappa_t


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



def log_accuracy_from_df(data, run_name, directory, merge_key, full):
    #print(data.columns)
    print(run_name)
    with open(f"{run_name}_acc_{full}_{version}.txt", "w") as f:
        if 'overall_accuracy' in data.columns:
            system_accuracy = data.iloc[-1]['overall_accuracy']
            if float(system_accuracy) < 1:
                system_accuracy = float(system_accuracy) * 100
            print(system_accuracy)
            f.write(f"Overall accuracy: {system_accuracy}\n")
            drift_point_accuracy, drift_point_km, drift_point_kt = get_drift_point_accuracy(data)
            detect_point_accuracy, detect_point_km, detect_point_kt = get_detect_point_accuracy(data)
            f.write(f"drift close accuracy: {drift_point_accuracy}\n")
            f.write(f"detect close accuracy: {detect_point_accuracy}\n")
        average_reuse_purity = 0
        overall_results = {}
        overall_results['Max Recall'] = 0
        overall_results['Max Precision'] = 0
        overall_results['Precision for Max Recall'] = 0
        overall_results['Recall for Max Precision'] = 0
        overall_results['f1'] = 0
        overall_results['MR by System'] = 0
        overall_results['MP by System'] = 0
        overall_results['PMR by System'] = 0
        overall_results['RMP by System'] = 0
        overall_results['f1 by System'] = 0
        overall_results['Num Good System Concepts'] = 0
        overall_results['Max Recall_alt'] = 0
        overall_results['Max Precision_alt'] = 0
        overall_results['Precision for Max Recall_alt'] = 0
        overall_results['Recall for Max Precision_alt'] = 0
        overall_results['f1_alt'] = 0
        overall_results['MR by System_alt'] = 0
        overall_results['MP by System_alt'] = 0
        overall_results['PMR by System_alt'] = 0
        overall_results['RMP by System_alt'] = 0
        overall_results['f1 by System_alt'] = 0
        overall_results['Num Good System Concepts_alt'] = 0
        if all(i in data.columns for i in ['ground_truth_concept', 'system_concept']):
            average_reuse_purity, concept_purity = get_concept_accuracy_from_df(data, merge_key)
            gt_results, overall_results_a = get_concept_transparency_data(data['ground_truth_concept'].values , data['system_concept'].values , concept_purity, merge_key)

            for k in overall_results_a.keys():
                overall_results[k] = overall_results_a[k]

            if 'alt_system_concept' in data:
                gt_results_alt, overall_results_alt = get_concept_transparency_data(data['ground_truth_concept'].values , data['alt_system_concept'].values , concept_purity, merge_key)
                #print(average_reuse_purity)
                for k in overall_results_alt.keys():
                    val = overall_results_alt[k]
                    new_key = f"{k}_alt"
                    overall_results[new_key] = val
            f.write(f"System Concept accuracy: {average_reuse_purity}\n")
            for k in overall_results.keys():
                f.write(f"{k}: {overall_results[k]}\n")
        drift_accuracy_50, drift_km_50, drift_kt_50 = get_drift_point_accuracy(data, 50)
        drift_accuracy_100, drift_km_100, drift_kt_100 = get_drift_point_accuracy(data, 100)
        drift_accuracy_250, drift_km_250, drift_kt_250 = get_drift_point_accuracy(data, 250)
        drift_accuracy_500, drift_km_500, drift_kt_500 = get_drift_point_accuracy(data, 500)
        drift_accuracy_750, drift_km_750, drift_kt_750 = get_drift_point_accuracy(data, 750)
        drift_accuracy_1000, drift_km_1000, drift_kt_1000 = get_drift_point_accuracy(data, 1000)
        detect_accuracy_50, detect_km_50, detect_kt_50 = get_detect_point_accuracy(data, 50)
        detect_accuracy_100, detect_km_100, detect_kt_100 = get_detect_point_accuracy(data, 100)
        detect_accuracy_250, detect_km_250, detect_kt_250 = get_detect_point_accuracy(data, 250)
        detect_accuracy_500, detect_km_500, detect_kt_500 = get_detect_point_accuracy(data, 500)
        detect_accuracy_750, detect_km_750, detect_kt_750 = get_detect_point_accuracy(data, 750)
        detect_accuracy_1000, detect_km_1000, detect_kt_1000 = get_detect_point_accuracy(data, 1000)
        uniquedetect_accuracy_50, uniquedetect_km_50, uniquedetect_kt_50 = get_uniquedetect_point_accuracy(data, 50)
        uniquedetect_accuracy_100, uniquedetect_km_100, uniquedetect_kt_100 = get_uniquedetect_point_accuracy(data, 100)
        uniquedetect_accuracy_250, uniquedetect_km_250, uniquedetect_kt_250 = get_uniquedetect_point_accuracy(data, 250)
        uniquedetect_accuracy_500, uniquedetect_km_500, uniquedetect_kt_500 = get_uniquedetect_point_accuracy(data, 500)
        uniquedetect_accuracy_750, uniquedetect_km_750, uniquedetect_kt_750 = get_uniquedetect_point_accuracy(data, 750)
        uniquedetect_accuracy_1000, uniquedetect_km_1000, uniquedetect_kt_1000 = get_uniquedetect_point_accuracy(data, 1000)
        driftdetect_accuracy_50, driftdetect_km_50, driftdetect_kt_50 = get_driftdetect_point_accuracy(data, 50)
        driftdetect_accuracy_100, driftdetect_km_100, driftdetect_kt_100 = get_driftdetect_point_accuracy(data, 100)
        driftdetect_accuracy_250, driftdetect_km_250, driftdetect_kt_250 = get_driftdetect_point_accuracy(data, 250)
        driftdetect_accuracy_500, driftdetect_km_500, driftdetect_kt_500 = get_driftdetect_point_accuracy(data, 500)
        driftdetect_accuracy_750, driftdetect_km_750, driftdetect_kt_750 = get_driftdetect_point_accuracy(data, 750)
        driftdetect_accuracy_1000, driftdetect_km_1000, driftdetect_kt_1000 = get_driftdetect_point_accuracy(data, 1000)
        ret_val = {
            'accuracy': system_accuracy,
            'drift_accuracy_50': drift_accuracy_50,
            'drift_km_50': drift_km_50,
            'drift_kt_50': drift_kt_50,
            'drift_accuracy_100': drift_accuracy_100,
            'drift_km_100': drift_km_100,
            'drift_kt_100': drift_kt_100,
            'drift_accuracy_250': drift_accuracy_250,
            'drift_km_250': drift_km_250,
            'drift_kt_250': drift_kt_250,
            'drift_accuracy_500': drift_accuracy_500,
            'drift_km_500': drift_km_500,
            'drift_kt_500': drift_kt_500,
            'drift_accuracy_750': drift_accuracy_750,
            'drift_km_750': drift_km_750,
            'drift_kt_750': drift_kt_750,
            'drift_accuracy_1000': drift_accuracy_1000,
            'drift_km_1000': drift_km_1000,
            'drift_kt_1000': drift_kt_1000,
            'detect_accuracy_50': detect_accuracy_50,
            'detect_km_50': detect_km_50,
            'detect_kt_50': detect_kt_50,
            'detect_accuracy_100': detect_accuracy_100,
            'detect_km_100': detect_km_100,
            'detect_kt_100': detect_kt_100,
            'detect_accuracy_250': detect_accuracy_250,
            'detect_km_250': detect_km_250,
            'detect_kt_250': detect_kt_250,
            'detect_accuracy_500': detect_accuracy_500,
            'detect_km_500': detect_km_500,
            'detect_kt_500': detect_kt_500,
            'detect_accuracy_750': detect_accuracy_750,
            'detect_km_750': detect_km_750,
            'detect_kt_750': detect_kt_750,
            'detect_accuracy_1000': detect_accuracy_1000,
            'detect_km_1000': detect_km_1000,
            'detect_kt_1000': detect_kt_1000,
            'uniquedetect_accuracy_50': uniquedetect_accuracy_50,
            'uniquedetect_km_50': uniquedetect_km_50,
            'uniquedetect_kt_50': uniquedetect_kt_50,
            'uniquedetect_accuracy_100': uniquedetect_accuracy_100,
            'uniquedetect_km_100': uniquedetect_km_100,
            'uniquedetect_kt_100': uniquedetect_kt_100,
            'uniquedetect_accuracy_250': uniquedetect_accuracy_250,
            'uniquedetect_km_250': uniquedetect_km_250,
            'uniquedetect_kt_250': uniquedetect_kt_250,
            'uniquedetect_accuracy_500': uniquedetect_accuracy_500,
            'uniquedetect_km_500': uniquedetect_km_500,
            'uniquedetect_kt_500': uniquedetect_kt_500,
            'uniquedetect_accuracy_750': uniquedetect_accuracy_750,
            'uniquedetect_km_750': uniquedetect_km_750,
            'uniquedetect_kt_750': uniquedetect_kt_750,
            'uniquedetect_accuracy_1000': uniquedetect_accuracy_1000,
            'uniquedetect_km_1000': uniquedetect_km_1000,
            'uniquedetect_kt_1000': uniquedetect_kt_1000,
            'driftdetect_accuracy_50': driftdetect_accuracy_50,
            'driftdetect_km_50': driftdetect_km_50,
            'driftdetect_kt_50': driftdetect_kt_50,
            'driftdetect_accuracy_100': driftdetect_accuracy_100,
            'driftdetect_km_100': driftdetect_km_100,
            'driftdetect_kt_100': driftdetect_kt_100,
            'driftdetect_accuracy_250': driftdetect_accuracy_250,
            'driftdetect_km_250': driftdetect_km_250,
            'driftdetect_kt_250': driftdetect_kt_250,
            'driftdetect_accuracy_500': driftdetect_accuracy_500,
            'driftdetect_km_500': driftdetect_km_500,
            'driftdetect_kt_500': driftdetect_kt_500,
            'driftdetect_accuracy_750': driftdetect_accuracy_750,
            'driftdetect_km_750': driftdetect_km_750,
            'driftdetect_kt_750': driftdetect_kt_750,
            'driftdetect_accuracy_1000': driftdetect_accuracy_1000,
            'driftdetect_km_1000': driftdetect_km_1000,
            'driftdetect_kt_1000': driftdetect_kt_1000,
            # 'drift_accuracy_100': get_drift_point_accuracy(data, 100),
            # 'drift_accuracy_250': get_drift_point_accuracy(data, 250),
            # 'drift_accuracy_500': get_drift_point_accuracy(data, 500),
            # 'drift_accuracy_750': get_drift_point_accuracy(data, 750),
            # 'drift_accuracy_1000': get_drift_point_accuracy(data, 1000),
            # 'detect_accuracy_50': get_detect_point_accuracy(data, 50),
            # 'detect_accuracy_100': get_detect_point_accuracy(data, 100),
            # 'detect_accuracy_250': get_detect_point_accuracy(data, 250),
            # 'detect_accuracy_500': get_detect_point_accuracy(data, 500),
            # 'detect_accuracy_750': get_detect_point_accuracy(data, 750),
            # 'detect_accuracy_1000': get_detect_point_accuracy(data, 1000),
            # 'uniquedetect_accuracy_50': get_uniquedetect_point_accuracy(data, 50),
            # 'uniquedetect_accuracy_100': get_uniquedetect_point_accuracy(data, 100),
            # 'uniquedetect_accuracy_250': get_uniquedetect_point_accuracy(data, 250),
            # 'uniquedetect_accuracy_500': get_uniquedetect_point_accuracy(data, 500),
            # 'uniquedetect_accuracy_750': get_uniquedetect_point_accuracy(data, 750),
            # 'uniquedetect_accuracy_1000': get_uniquedetect_point_accuracy(data, 1000),
            # 'driftdetect_accuracy_50': get_driftdetect_point_accuracy(data, 50),
            # 'driftdetect_accuracy_100': get_driftdetect_point_accuracy(data, 100),
            # 'driftdetect_accuracy_250': get_driftdetect_point_accuracy(data, 250),
            # 'driftdetect_accuracy_500': get_driftdetect_point_accuracy(data, 500),
            # 'driftdetect_accuracy_750': get_driftdetect_point_accuracy(data, 750),
            # 'driftdetect_accuracy_1000': get_driftdetect_point_accuracy(data, 1000),
            # 'detect accuracy': detect_point_accuracy,
            'average_purity': average_reuse_purity,
        }
        for k in ret_val.keys():
            f.write(f"{k}: {ret_val[k]}\n")
    for k in overall_results.keys():
        ret_val[k] = overall_results[k]
    

    # with open(f"{run_name}_result_{version}_{full}.pickle", "wb") as f:
    #     pickle.dump(ret_val, f)

    return ret_val

def plot_outerFSM(dataframes, run_names, directory):
    transitions = {}
    totals = {}
    last_concept = None
    for i, row in dataframes[0].iterrows():
        concept = row['system_concept']
        if last_concept is None or concept != last_concept:
            if not concept in transitions:
                transitions[concept] = {}
                totals[concept] = 0
            if not last_concept is None:
                if not concept in transitions[last_concept]:
                    transitions[last_concept][concept] = 0
                transitions[last_concept][concept] += 1
                totals[last_concept]+= 1
            last_concept = concept
    print(transitions)

    dot = Digraph(comment="FSM")
    dot.graph_attr['rankdir'] = 'LR'
    for node_id in transitions:
        print(str(node_id))
        dot.node(str(node_id), str(node_id))
        

    for from_id in transitions:
        to_ids_counter = transitions[from_id]
        for to_id in to_ids_counter:
            dot.edge(str(from_id), str(to_id), str(round(to_ids_counter[to_id] / totals[from_id] * 100) / 100))

    dot.render(f'{directory}\{run_names[0]}_FSM.gv', view= False)

def plot_system_acc_from_df(dataframes, run_names, directory):
    final_ordering = []
    for i, df in enumerate(dataframes):
        if 'sliding_window_accuracy' in df.columns:
            sns.lineplot(x='example', y='sliding_window_accuracy', data=df, size=1, legend=None)
            final_ordering.append(run_names[i])
    plt.legend(final_ordering)
    plt.savefig(f"{directory}{os.sep}{run_names[0]}_sys_accuracy.png")
    plt.clf()


def plot_concepts_from_df(data, run_name, directory): 
    # Iterate over possible states
    plotting_states = True  
    state_id = 0
    plt.figure(figsize=(20,5))
    while plotting_states:
        state_column_name = f'state_{state_id}_acc'
        if state_column_name in data.columns:
            sns.lineplot(x='example', y=state_column_name,
                linewidth=1, data=data,
                color = sns.color_palette()[state_id % len(sns.color_palette())],
                legend=None,
                hue_order=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        else:
            plotting_states = False
        state_id += 1
    if 'overall_accuracy' in data.columns:
        data['overall_accuracy'] = data['overall_accuracy']/100
        data['sliding_window_accuracy'] = data['sliding_window_accuracy']/100
        sns.lineplot(x='example', y='overall_accuracy',
                linewidth=1, data=data,
                color = 'black',
                legend=None)
        sns.lineplot(x='example', y='sliding_window_accuracy',
                linewidth=1, data=data,
                color = 'red',
                legend=None)
    if 'change_detected' in data.columns:
        # Plot the change detections in green
        for i, row in data[data['change_detected'] != 0].iterrows():
            plt.plot([row['example'], row['example']], [0.08, 0.12], color = "green")

    if 'ground_truth_concept' in data.columns:
        lag_row = pd.Series()
        start_row = pd.Series()
        for i, row in data.iterrows():
            if 'ground_truth_concept' in start_row:
                if row['ground_truth_concept'] != start_row['ground_truth_concept']:
                    plt.plot([start_row['example'], row['example']], [0.05, 0.05], 
                        color = sns.color_palette()[int(start_row['ground_truth_concept']) % len(sns.color_palette())])
                    lag_row = start_row
                    start_row = row
            else:
                lag_row = start_row
                start_row = row
        last_example_estimate = data['example'].iloc[-1] + (data['example'].iloc[-1] - data['example'].iloc[-2])
        plt.plot([start_row['example'], last_example_estimate], [0.05, 0.05], 
                        color = sns.color_palette()[int(start_row['ground_truth_concept']) % len(sns.color_palette())])        


    if 'model_update' in data.columns:
        # Plot model updates (hoeffding tree splits)
        for row in data[data['model_update'] != 0]:
            plt.plot([row['example'], row['example']], [0.5, 0.7])

    if 'drift_occured' in data.columns:
        # Plot model updates (hoeffding tree splits)
        for i,row in data[data['drift_occured'] != 0].iterrows():
            plt.plot([row['example'], row['example']], [0.03, 0.07], color="red")  

    if 'system_concept' in data.columns:
        lag_row = pd.Series()
        start_row = pd.Series()
        for i, row in data.iterrows():
            if 'system_concept' in start_row:
                if row['system_concept'] != start_row['system_concept']:
                    plt.plot([start_row['example'], row['example']], [0.1, 0.1], 
                        color = sns.color_palette()[int(start_row['system_concept']) % len(sns.color_palette())])
                    lag_row = start_row
                    start_row = row
            else:
                lag_row = start_row
                start_row = row
        last_example_estimate = data['example'].iloc[-1] + (data['example'].iloc[-1] - data['example'].iloc[-2])
        plt.plot([start_row['example'], last_example_estimate], [0.1, 0.1], 
                        color = sns.color_palette()[int(start_row['system_concept']) % len(sns.color_palette())])  

    plt.savefig(f"{directory}{os.sep}{run_name}.pdf")
    plt.clf()
# gt = np.array([1,1,1,1,2,2,2,2])
# sys = np.array([1,1,2,2,2,3,3,3])
# rv, pv = get_concept_transparency_data(gt, sys, np.array([1,1,1,1,1,1,1,1]), None)
# print(rv)
# print(pv)
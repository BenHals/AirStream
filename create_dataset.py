
import pathlib
import json
import math
import argparse

import utm
import numpy as np
import pandas as pd






def create_dataset(raw_data_directory, data_name, target_sensor_index, output_directory, direction = True, seed = 5000, broken_proportion = 0.02, broken_length = 75):
    rand_state = np.random.RandomState(seed)

    full_data_path = raw_data_directory / f"{data_name}_full.csv"
    print(full_data_path)
    sensor_path = raw_data_directory / f"{data_name}_sensors.json"
    if not raw_data_directory.exists():
        raise ValueError("No Directory")
    if not full_data_path.exists():
        raise ValueError("full data, process data")
    if not sensor_path.exists():
        raise ValueError("No sensors, process data")
    print("data is fine")



    with sensor_path.open() as f:
        sensor_locations = json.load(f)

    full_data = pd.read_csv(full_data_path)
    print(full_data['date_time'])
    full_data['date_time'] = pd.to_datetime(full_data['date_time'])
    print(full_data['date_time'])
    # print(full_data['date_time'].dt.month)
    # exit()
    # aux_data["date_time"] = pd.to_datetime(aux_data["date_time"])
    full_data['s'] = full_data["date_time"].dt.second
    full_data['m'] = full_data["date_time"].dt.minute
    full_data['h'] = full_data["date_time"].dt.hour
    full_data['day'] = (full_data["date_time"].dt.hour > 10) & (full_data["date_time"].dt.hour < 17)
    full_data['wd'] = full_data["date_time"].dt.weekday
    full_data['we'] = full_data["date_time"].dt.weekday >= 5

    aux_cols = [c for c in full_data.columns if c not in sensor_locations]

    target_sensor = [k for k in sensor_locations][target_sensor_index]
    lag_amount = 1
    dataset_columns = [k for k in sensor_locations if k != target_sensor]
    for l in range(1, lag_amount+1):
        for sensor_id in [k for k in sensor_locations if k != target_sensor]:
            col_name = f"{sensor_id}_lag_{l}"
            dataset_columns.append(col_name)
            full_data[col_name] = full_data[sensor_id].shift(l)
    # for l in reversed(range(1, lag_amount+1)):
    #     sensor_id = target_sensor
    #     col_name = f"{sensor_id}_lag_{l}"
    #     dataset_columns.append(col_name)
    #     full_data[col_name] = full_data[sensor_id].shift(l)

    bins = [12, 35.4, 55.4, 150.4, 250.4]
    # bins = [6, 17.4, 27.7, 75.2, 250.4]
    # bins = [9, 25, 42, 100, 250.4]
    # bins = [0.25, 0.75, 2, 5, 10]
    # vals, bins = pd.qcut(full_data[target_sensor], 6, retbins =True, labels = False)
    print(bins)
    bins = list(bins)
    # for l in reversed(range(1, lag_amount+1)):
    #     sensor_id = target_sensor
    #     col_name = f"{sensor_id}_lagL_{l}"
    #     dataset_columns.append(col_name)
    #     dataset[col_name] = pd.np.digitize(full_data[sensor_id].shift(l), bins)
    if not direction:
        full_data['y'] = pd.np.digitize(full_data[target_sensor], bins)
    else:
        diff = full_data[target_sensor] - full_data[sensor_id].shift(1)
        vals, bins = pd.qcut(diff, 6, retbins =True, labels = False, duplicates = "drop")
        bins = list(bins)
        full_data['y'] = pd.np.digitize(diff, bins)
    dataset_columns.append("y")
    full_data = full_data.dropna()

    aux_data = full_data[aux_cols]
    dataset = full_data[dataset_columns]

    # mask = rand_state.choice(2, dataset.shape[0], p = [0.95, 0.05])
    # broken_chance = 0.03
    # broken_chance = 0.02
    broken_chance = broken_proportion
    if broken_proportion > 0: 
        # mask = rand_state.choice(2, dataset.shape[0], p = [1 - broken_chance, broken_chance])
        num_mask_blocks = (dataset.shape[0] * broken_proportion) // broken_length
        non_mask_observations = (dataset.shape[0] * (1-broken_proportion))
        non_mask_block_size = non_mask_observations // (num_mask_blocks + 1)
        mask = []
        next_mask_block = 0 + (rand_state.randint(non_mask_block_size * 0.50, non_mask_block_size * 1.50))
        for i in range(dataset.shape[0]):
            if i < next_mask_block:
                mask.append(0)
            elif i < next_mask_block + broken_length:
                mask.append(1)
            else:
                next_mask_block = next_mask_block + broken_length + (rand_state.randint(non_mask_block_size * 0.75, non_mask_block_size * 1.25))
                mask.append(0)
    else:
        mask = []
        for i in range(dataset.shape[0]):
            mask.append(0)

    print(mask)

    # mask = rand_state.choice(2, dataset.shape[0], p = [0.99, 0.01])
    dataset['mask'] = mask
    print(dataset['mask'])
    # for l in range(60):
    # broken_length = 75
    # broken_length = 60
    # broken_length = 90
    # for l in range(broken_length):
    # # for l in range(90):
    # # for l in range(24):
    #     dataset['mask'] = dataset['mask'] | dataset['mask'].shift(1).fillna(0).astype(int)
    #     print(dataset['mask'])
    # print(dataset['mask'])

    proportion_mask = dataset['mask'].sum() / dataset['mask'].shape[0]
    print(proportion_mask)
    # exit()
    # aux_data = aux_data.loc[dataset.index]

    # In meters per second
    avg_windspeed = 2

    # In seconds
    time_period = 60 * 60
    temporal_distance = avg_windspeed * time_period
    spacial_pattern = []
    target_x = sensor_locations[target_sensor][0] or temporal_distance
    target_y = sensor_locations[target_sensor][1] or temporal_distance
    target_t = 0
    supp_info = {'sensors': []}
    for l in range(lag_amount + 1):
        for sensor in sensor_locations:
            if sensor == target_sensor:
                continue
            s_x = sensor_locations[sensor][0] or temporal_distance * 1.1
            s_y = sensor_locations[sensor][1] or temporal_distance * 1.1
            s_t = l * temporal_distance
            distance = math.sqrt(math.pow(s_x - target_x, 2) + math.pow(s_y - target_y, 2) + math.pow(s_t - target_t, 2))
            spacial_pattern.append((s_x, s_y, s_t, distance))

    # for l in reversed(range(1, lag_amount+1)):
    #     s_x = target_x
    #     s_y = target_y
    #     s_t = l * temporal_distance
    #     distance = math.sqrt(math.pow(s_x - target_x, 2) + math.pow(s_y - target_y, 2) + math.pow(s_t - target_t, 2))
    #     spacial_pattern.append((s_x, s_y, s_t, distance))
    for l in reversed(range(1, lag_amount+1)):
        s_x = target_x or -2
        s_y = target_y or -2
        s_t = l * temporal_distance
        distance = math.sqrt(math.pow(s_x - target_x, 2) + math.pow(s_y - target_y, 2) + math.pow(s_t - target_t, 2))
        spacial_pattern.append((s_x, s_y, s_t, distance))
    supp_info['sensors'] = spacial_pattern
    # target_sensor = [k for k in sensor_locations][target_sensor_index]
    # lag_amount = 1
    # dataset_columns = [k for k in sensor_locations if k != target_sensor]
    # dataset = full_data[dataset_columns]
    # for l in range(1, lag_amount+1):
    #     for sensor_id in [k for k in sensor_locations if k != target_sensor]:
    #         col_name = f"{sensor_id}_lag_{l}"
    #         dataset_columns.append(col_name)
    #         dataset[col_name] = dataset[sensor_id].shift(l)

    # # bins = [12, 35.4, 55.4, 150.4, 250.4]
    # # bins = [0.25, 0.75, 2, 5, 10]
    # vals, bins = pd.qcut(full_data[target_sensor], 12, retbins =True, labels = False)
    # print(vals)
    # print(bins)
    # bins = list(bins)
    # for l in range(1, lag_amount+1):
    #     sensor_id = target_sensor
    #     col_name = f"{sensor_id}_lagL_{l}"
    #     dataset_columns.append(col_name)
    #     dataset[col_name] = pd.np.digitize(full_data[sensor_id].shift(l), bins)
    # for l in range(1, lag_amount+1):
    #     sensor_id = target_sensor
    #     col_name = f"{sensor_id}_lag_{l}"
    #     dataset_columns.append(col_name)
    #     dataset[col_name] = full_data[sensor_id].shift(l)
    # dataset['y'] = pd.np.digitize(full_data[target_sensor], bins)

    # # In meters per second
    # avg_windspeed = 2

    # # In seconds
    # time_period = 60 * 60
    # temporal_distance = avg_windspeed * time_period
    # spacial_pattern = []
    # target_x = sensor_locations[target_sensor][0]
    # target_y = sensor_locations[target_sensor][1]
    # target_t = 0
    # supp_info = {'sensors': []}
    # for l in range(lag_amount + 1):
    #     for sensor in sensor_locations:
    #         if sensor == target_sensor:
    #             continue
    #         s_x = sensor_locations[sensor][0]
    #         s_y = sensor_locations[sensor][1]
    #         s_t = l * temporal_distance
    #         distance = math.sqrt(math.pow(s_x - target_x, 2) + math.pow(s_y - target_y, 2) + math.pow(s_t - target_t, 2))
    #         spacial_pattern.append((s_x, s_y, s_t, distance))


    # for l in range(1, lag_amount + 1):
    #     s_x = target_x
    #     s_y = target_y
    #     s_t = l * temporal_distance
    #     distance = math.sqrt(math.pow(s_x - target_x, 2) + math.pow(s_y - target_y, 2) + math.pow(s_t - target_t, 2))
    #     spacial_pattern.append((s_x, s_y, s_t, distance))
    # for l in range(1, lag_amount + 1):
    #     s_x = target_x
    #     s_y = target_y
    #     s_t = l * temporal_distance
    #     distance = math.sqrt(math.pow(s_x - target_x, 2) + math.pow(s_y - target_y, 2) + math.pow(s_t - target_t, 2))
    #     spacial_pattern.append((s_x, s_y, s_t, distance))
    # supp_info['sensors'] = spacial_pattern

    output_directory = pathlib.Path(output_directory) / data_name / str(target_sensor_index) / str(seed)
    output_directory.mkdir(parents= True, exist_ok=True)

    data_fn = output_directory / f"stream-{data_name}_dataset.csv"
    aux_fn = f"{output_directory / data_name}_aux.csv"
    info_fn = output_directory / f"df_info.json"
    time_index_fn = output_directory / f"time_index.pickle"
    dist_fn = output_directory / f"dist.json"

    full_data['date_time'].to_pickle(time_index_fn)
    dataset.to_csv(data_fn, index = False)
    aux_data.to_csv(aux_fn, index = False)
    with open(output_directory / f"spacial_pattern.json", 'w') as f:
        json.dump(spacial_pattern, f)
    with open(info_fn, 'w') as f:
        json.dump({"name": data_name, "target_index": target_sensor_index, "target_sensor": target_sensor, "bins": bins, "direction": direction, 'mask_proportion': proportion_mask, 'broken_chance': broken_chance, 'broken_length': broken_length}, f)
    with open(dist_fn, 'w') as f:
        json.dump(str(dataset["y"].value_counts()), f)
    return output_directory, data_fn, aux_fn, info_fn, time_index_fn



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str, help="The directory containing the experiment")
    parser.add_argument("-od", default= None, type=str, help="The directory containing the experiment")
    parser.add_argument("-n", default = "Rangiora", type=str, help="name")
    parser.add_argument("-ti", type=int, default = 0, help="target index")

    # parser.add_argument("-lf", default=f"experiment-{time.time()}.log", type=str, help="The name of the file to log to")

    args = parser.parse_args()
    if args.od is None:
        args.od = pathlib.Path.cwd() / 'experiments'
    create_dataset(pathlib.Path.cwd() / args.d, args.n, args.ti, args.od)


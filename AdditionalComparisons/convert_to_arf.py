import pandas as pd
import numpy as np
import pathlib
import argparse
import tqdm

def process_csv_with_mask(fp):
    df = pd.read_csv(fp)
    if 'mask' in df.columns:
        df['mask'] = df['mask'].astype(int)
    else:
        df['mask'] = np.zeros(df.shape[0]).astype(int)
        df = df.drop(columns=[df.columns[0]])
        new_cols = [*[f"f{x}" for x,_ in enumerate(df.columns[:-2])], 'y', 'mask']
        df.columns = new_cols
    new_cols = [*df.columns[:-2], df.columns[-1], df.columns[-2]]
    df = df[new_cols]
    print(df)
    # print(df['y'])
    # exit()
    new_fp = pathlib.Path(fp.parent) / f"{fp.stem}.arff"
    print(new_fp)
    with open(new_fp, 'w') as f:
        print("@relation data", file=f)
        for c in df.columns[:-2]:
            print(f"@attribute {c} numeric", file=f)
        print("@attribute mask {0,1}", file=f)
        print(f"@attribute y {{{','.join([str(x) for x in df['y'].unique()])}}}", file=f)
        print("", file=f)
        print("@data", file=f)
        for row in tqdm.tqdm(df.itertuples(index=False)):
            csv_row = ','.join([str(x) for x in row])
            print(csv_row, file=f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", default="cwd", type=str, help="The directory containing the experiment")
    parser.add_argument("-o", default="cwd", type=str, help="The directory containing the output")
    parser.add_argument("-m", default="*", type=str, help="string to match files")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.d)
    output_dir = pathlib.Path(args.o)
    input_files = list(input_dir.rglob(args.m))

    for fp in input_files:
        process_csv_with_mask(fp)

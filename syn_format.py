#%%
import pathlib
import pickle
import numpy as np
import pandas as pd

df_res = pd.read_pickle("syn_res.pickle")


#%%
df_res
#%%
df_res['ks_detect'] = df_res['ks_detect'] * 100

# %%
df_res.unstack('generator')

# %%
df_stack = df_res.stack(0)

# %%
df_stack['res'] = df_stack['mean'].apply(lambda x : f"{x/100:.2f}").astype(str) + ' ('+  df_stack['std'].apply(lambda x : f"{x/100:.2f}").astype(str) + ')'
# df_stack['res'] = ((df_stack['mean']).round() / 100).astype(str) + ' ('+  ((df_stack['std']).round() / 100).astype(str) + ')'
# df_stack[['mean', 'std']].apply(lambda x: f"{x['mean']} ({x['std']})")
# %%
df_stack

# %%
df_2 = df_stack['res'].unstack('generator')

# %%
df_2 = df_2.unstack(1)
# df_2.unstack('2').reorder_levels([1, 0], axis = 1)

# %%
df_2

# %%
df_2.index

# %%
df_2 = df_2.reindex(['ks_detect', 'f1', 'time', 'mem'], axis = 1, level = 1)

# %%
df_2 = df_2.reindex(['NC', 'OK', 'RF', 'ARF', 'AirStreamNoBacktrack', 'AirStreamBacktrack'])

# %%
df_2

# %%

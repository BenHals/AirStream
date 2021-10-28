#%%
rbf_kappa = {
    ("NC", 14.26),
    ("OK", -1.75),
    ("RF", 7.75),
    ("ARF", 59.27),
    ("DYNSE", 36.68),
    ("AS_b", 59.44),
    ("AS_r", 59.59),
}
rbf_CF1 = {
    ("NC", 0.48),
    ("OK", 0.48),
    ("RF", 0.48),
    ("ARF", 0.48),
    ("DYNSE", 0.47),
    ("AS_b", 0.71),
    ("AS_r", 0.76),
}
rt_kappa = {
    ("NC", 1.14),
    ("OK", 0.05),
    ("RF", 1.43),
    ("ARF", 50.68),
    ("DYNSE", 38.30),
    ("AS_b", 53.27),
    ("AS_r", 54.76),
}
rt_CF1 = {
    ("NC", 0.48),
    ("OK", 0.48),
    ("RF", 0.48),
    ("ARF", 0.46),
    ("DYNSE", 0.51),
    ("AS_b", 0.63),
    ("AS_r", 0.71),
}

df = 25 * 4 * 9

for dataset, ds_name in [(rbf_kappa, 'rbf_kappa'), (rbf_CF1, 'rbf_CF1'), (rt_kappa, 'rt_kappa'), (rt_CF1, 'rt_CF1')]:
    sorted_means = sorted([x[1] for x in dataset], reverse=True)
    ranked = [(x[0], sorted_means.index(x[1]) + 1) for x in dataset]
    print(ranked)
    from Orange.evaluation import compute_CD, graph_ranks
    means = [x if not np.isnan(x) else 0 for x in dataset]
    print(means)
    print(r.shape[0])
    # cd = compute_CD(means, r.shape[0], test = 'bonferroni-dunn')
    cd = compute_CD(means, r.shape[0])
    print(cd)
    graph_ranks(means, systems_to_compare, cd, width = 10, textspace = 3, filename = f"{ds_name}_v2")
# %%

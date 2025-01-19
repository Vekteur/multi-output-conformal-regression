import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from moc.utils.general import savefig
from moc.analysis.helpers import get_metric_name, conformal_methods
from moc.analysis.dataframes import get_datasets_df


# Plot the data conditional to X
def plot_data_conditional(X, Y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y[:, 0], Y[:, 1], c='b', marker='o')
    ax.set(xlabel='$X$', ylabel='$Y_1$', zlabel='$Y_2$')
    plt.show()


# Plot the data ignoring X
def plot_data_unconditional(Y):
    fig, ax = plt.subplots()
    ax.scatter(Y[:, 0], Y[:, 1], c='b', marker='o')
    ax.set(xlabel='$Y_1$', ylabel='$Y_2$')
    plt.show()


# Plot the grid in the unit ball
def plot_grid(g):
    _, ax = plt.subplots()
    ax.scatter(g[:, 0], g[:, 1], c='r', marker='o')
    plt.show()


def plot_coverage(ax, plot_df, posthoc_methods, alpha, palette):
    posthoc_methods = [m for m in posthoc_methods if m in plot_df.index.get_level_values('posthoc_method').unique()]
    g = sns.barplot(
        plot_df, 
        x='dataset', 
        y='value', 
        hue='posthoc_method', 
        order=plot_df.index.get_level_values('dataset').unique(), 
        hue_order=posthoc_methods,
        errorbar=('se', 1),
        capsize=0.1,
        err_kws={'linewidth': 1},
        palette=palette,
        ax=ax,
    )
    ax.tick_params(axis='x', which='major', labelsize=7, labelrotation=90)
    ax.set(xlabel='', ylabel='Coverage', ylim=(0, 1))
    g.legend().remove()# set_title(None)
    ax.axhline(1 - alpha, color='black', linestyle='--')


def plot_coverage_per_model(plot_df, posthoc_methods, alpha, palette, path):
    models = plot_df.index.get_level_values('model').unique()
    fig, ax = plt.subplots(len(models), 1, figsize=(12, 4), sharex=True, sharey=True, squeeze=False)
    ax = ax.flatten()
    for axis, (model_name, model_df) in zip(ax, plot_df.groupby('model')):
        print(model_name)
        plot_coverage(axis, model_df, posthoc_methods, alpha, palette)
        if model_name == 'MQF2':
            handles, labels = axis.get_legend_handles_labels()
        axis.set_ylabel(f'Coverage of \n{model_name}', fontsize=8)

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.9),
        frameon=True,
        ncol=len(posthoc_methods),
    )
    savefig(path)


def plot_n_samples_all_methods(df, config):
    metrics = ['log_region_size', 'cond_cov_x_error', 'cond_cov_z_error', 'wsc']
    hparams = {
        'PCP': ('posthoc_n_samples', '$L$'),
        'HD-PCP': ('posthoc_n_samples', '$L$'),
        'C-PCP': ('posthoc_n_samples_mc', '$K, L$'),
        'HDR-CP': ('posthoc_n_samples', '$K$'),
    }

    nrows = len(metrics)
    ncols = len(hparams)

    groupby = set(df.index.names) - {'run_id'}
    plot_df = df.groupby(list(groupby), dropna=False, observed=True).mean().reset_index()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))

    for col, (method, (hparam, hparam_label)) in enumerate(hparams.items()):
        method_df = plot_df.query('posthoc_method == @method')
        for row, metric in enumerate(metrics):
            axis = axes[row, col]
            metric_df = method_df.query('metric == @metric')
            for dataset, ds_df in metric_df.groupby('dataset', dropna=False, observed=True):
                n_samples = [10, 30, 100, 300]
                ds_df = ds_df.query(f'{hparam} in @n_samples')
                ds_df = ds_df.sort_values(hparam)
                ds_df[hparam] = ds_df[hparam].astype(int).astype(str)
                # If the metric is log_region_size, normalize such that the minimium value is 0 and the maximum value is 1
                if metric == 'log_region_size':
                    min_value, max_value = ds_df['value'].min(), ds_df['value'].max()
                    value = (ds_df['value'] - min_value) / (max_value - min_value)
                else:
                    value = ds_df['value']
                axis.plot(ds_df[hparam], value, 'o-', label=dataset)
                # if len(ds_df) > 0:
                #     display(ds_df)
            if metric in ['cond_cov_x_error', 'cond_cov_z_error']:
                axis.set_yscale('log')
            if metric == 'wsc':
                axis.axhline(1 - config.alpha, color='black', linestyle='--')
            axis.set_xlabel(hparam_label, fontsize=18)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1),
        frameon=True,
        ncol=5,
        fontsize=20,
        title_fontsize=14,
    )
    fig.tight_layout()


def plot_n_samples(df, config, method, datasets_subset=None, reg_line=False):
    df = df.query('posthoc_method == @method')
    #metrics = ['log_region_size', 'cond_cov_x_error', 'cond_cov_z_error', 'wsc']
    metrics = ['coverage', 'cond_cov_x_error', 'cond_cov_z_error', 'wsc']
    relative_metrics = ['log_region_size', 'cond_cov_x_error', 'cond_cov_z_error']
    hparams = {
        'PCP': ('posthoc_n_samples', '$L$'),
        'HD-PCP': ('posthoc_n_samples', '$L$'),
        'C-PCP': ('posthoc_n_samples_mc', '$K\ (=L)$'),
        'HDR-CP': ('posthoc_n_samples', '$K$'),
    }
    hparam, hparam_label = hparams[method]

    groupby = set(df.index.names) - {'run_id'}
    plot_df = df.groupby(list(groupby), dropna=False, observed=True).mean().reset_index()

    ncols = len(metrics)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 2.5))

    for col, metric in enumerate(metrics):
        metric_name = get_metric_name(metric)
        axis = axes[col]
        metric_df = plot_df.query('metric == @metric')

        n_samples = [10, 30, 100, 300]
        metric_df = metric_df.query(f'{hparam} in @n_samples')
        metric_df = metric_df.sort_values(hparam)
        metric_df[hparam] = metric_df[hparam].astype(int).astype(str)
        # Normalize such that the minimium value is 0 and the maximum value is 1
        if metric in relative_metrics:
            metric_df['value_to_plot'] = (metric_df
                .groupby('dataset', dropna=True, observed=True)['value']
                .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
            )
            metric_name = f'Relative {metric_name}'
        else:
            metric_df['value_to_plot'] = metric_df['value']
        subset_df = metric_df.query(f'dataset in @datasets_subset')
        subset_df['dataset'] = pd.Categorical(subset_df['dataset'], categories=datasets_subset, ordered=True)
        sns.lineplot(
            data=subset_df, x=hparam, y='value_to_plot', hue='dataset', hue_order=datasets_subset, style='dataset',
            palette='colorblind', markers=True, dashes=True, ax=axis
        )
        if reg_line:
            # Map [10, 30, 100, 300] to [0, 1, 2, 3]
            metric_df['new_x'] = metric_df[hparam].map({str(n): i for i, n in enumerate(n_samples)})
            sns.regplot(data=metric_df, x='new_x', y='value_to_plot', scatter=False, color='red', ax=axis)
        handles, labels = axis.get_legend_handles_labels()
        axis.legend_.remove()

        if metric in ['coverage', 'wsc']:
            axis.axhline(1 - config.alpha, color='black', linestyle='--')
            axis.set_ylim(0.35, 1)
        axis.set_xlabel(hparam_label, fontsize=17)
        axis.set_ylabel(metric_name, fontsize=17)

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1),
        frameon=False,
        ncol=11,
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def plot_ndim(df, config, dataset_start):
    groupby = set(df.index.names) - {'run_id'}
    plot_df = df.groupby(list(groupby), dropna=False, observed=True).mean().reset_index()
    plot_df = plot_df.query('model == "MQF2"')
    plot_df = plot_df.loc[plot_df['dataset'].str.startswith(dataset_start)]
    methods = conformal_methods #['C-PCP', 'HDR-CP', 'PCP', 'HD-PCP', 'DR-CP']
    plot_df = plot_df.query('posthoc_method in @methods')
    plot_df = plot_df.reset_index()
    # Merge plot_df and df_ds on column 'dataset'
    df_ds = get_datasets_df(config, reload=False)
    plot_df = plot_df.merge(df_ds.reset_index().rename(columns={'Dataset': 'dataset', 'Nb targets': 'd'}), on='dataset')
    plot_df['d'] = plot_df['d'].astype(str)
    plot_df['posthoc_method'] = pd.Categorical(plot_df['posthoc_method'], methods)

    metrics = ['log_region_size', 'cond_cov_x_error', 'cond_cov_z_error', 'wsc']
    ncols = len(metrics)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 2.5))

    for col, metric in enumerate(metrics):
        axis = axes[col]
        metric_df = plot_df.query('metric == @metric')
        sns.lineplot(data=metric_df, x='d', y='value', hue='posthoc_method', style='posthoc_method', markers=True, ax=axis)
        handles, labels = axis.get_legend_handles_labels()
        axis.legend_.remove()
        if metric == 'wsc':
            axis.axhline(1 - config.alpha, color='black', linestyle='--')
        metric_name = get_metric_name(metric)
        axis.set_xlabel('$d$', fontsize=17)
        axis.set_ylabel(metric_name, fontsize=17)

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.92),
        frameon=False,
        ncol=11,
        fontsize=14,
    )
    fig.tight_layout()

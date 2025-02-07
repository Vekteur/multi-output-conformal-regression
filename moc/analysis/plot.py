import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from moc.utils.general import savefig
from .helpers import get_metric_name, conformal_methods, main_metrics
from .dataframes import get_datasets_df
from .cmaps import get_cmap


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
        'C-HDR': ('posthoc_n_samples', '$K$'),
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


def plot_n_samples(df, config, method, datasets_subset=None, reg_line=False, ncols=3):
    df = df.query('posthoc_method == @method')
    #metrics = ['log_region_size', 'cond_cov_x_error', 'cond_cov_z_error', 'wsc']
    metrics = ['cond_cov_x_error', 'cond_cov_z_error', 'wsc', 'coverage', 'median_region_size', 'region_size']
    relative_metrics = ['cond_cov_x_error', 'cond_cov_z_error', 'median_region_size', 'region_size']
    hparams = {
        # 'PCP': ('posthoc_n_samples', '$L$'),
        # 'HD-PCP': ('posthoc_n_samples', '$L$'),
        'C-PCP': ('posthoc_n_samples_mc', '$K$'),
        'C-HDR': ('posthoc_n_samples', '$K$'),
    }
    hparam, hparam_label = hparams[method]

    df = df[['dataset', 'metric', hparam, 'value']]
    plot_df = df.groupby(['dataset', 'metric', hparam], dropna=False, observed=True).mean().reset_index()

    nrows = np.ceil(len(metrics) / ncols).astype(int)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 2.9))
    axes = axes.flatten()

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
        metric_df['value_to_plot'] = metric_df['value_to_plot'].astype(np.float32)
        subset_df = metric_df.query(f'dataset in @datasets_subset')
        subset_df['dataset'] = pd.Categorical(subset_df['dataset'], categories=list(datasets_subset), ordered=True)
        sns.lineplot(
            data=subset_df, x=hparam, y='value_to_plot', hue='dataset', hue_order=datasets_subset, style='dataset',
            palette='colorblind', markers=True, dashes=True, ax=axis
        )
        if reg_line:
            # Map [10, 30, 100, 300] to [0, 1, 2, 3]
            metric_df['new_x'] = metric_df[hparam].map({str(n): i for i, n in enumerate(n_samples)})
            sns.regplot(data=metric_df, x='new_x', y='value_to_plot', scatter=False, color='red', ax=axis, line_kws={'linewidth': 2})
        handles, labels = axis.get_legend_handles_labels()

        axis.get_legend().remove()

        if metric in ['coverage', 'wsc']:
            axis.axhline(1 - config.alpha, color='black', linestyle='--')
            axis.set_ylim(0.35, 1)
        axis.set_xlabel(hparam_label, fontsize=14)
        axis.set_ylabel(metric_name, fontsize=14)

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.97),
        frameon=False,
        ncol=7,
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def plot_ndim(df, config, dataset_start):
    groupby = set(df.index.names) - {'run_id'}
    plot_df = df.groupby(list(groupby), dropna=False, observed=True).mean().reset_index()
    plot_df = plot_df.query('model == "MQF2"')
    plot_df = plot_df.loc[plot_df['dataset'].str.startswith(dataset_start)]
    methods = conformal_methods #['C-PCP', 'C-HDR', 'PCP', 'HD-PCP', 'DR-CP']
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
        axis.get_legend().remove()
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


blue_palette = sns.color_palette("Blues", 5)  # Larger palette for better control
red_palette = sns.color_palette("YlOrRd", 4)
green_palette = sns.color_palette("Greens", 5)

color_map = {
    'M-CP': green_palette[2],  # Slightly darker green
    'CopulaCPTS': green_palette[4],
    'DR-CP': blue_palette[2],  # Slightly darker blue
    'C-HDR': blue_palette[3],
    'PCP': red_palette[0],
    'HD-PCP': red_palette[1],
    'STDQR': red_palette[2],
    'C-PCP': red_palette[3],
    'L-CP': 'violet',
    'HDR-H': 'gray',
    'L-H': 'gray',
}

style_map = {
    'M-CP': '-',
    'CopulaCPTS': '--',
    'DR-CP': '-',
    'C-HDR': '-',
    'PCP': '--',
    'HD-PCP': ':',
    'STDQR': ':',
    'C-PCP': '--',
    'L-CP': '-',
    'HDR-H': '-',
    'L-H': '--',
}


def plot_comparison(df, config, ax, metric, plot_type='pointplot', cmap=None):
    df = df.copy()
    names = df['name'].unique()
    df['name'] = pd.Categorical(df['name'], categories=[name for name in conformal_methods if name in names], ordered=True)
    cmap = get_cmap(df, cmap)
    
    df = df[['dataset', 'value', 'name', 'run_id']]

    size_metrics = ['region_size', 'median_region_size', 'exact_region_size', 'log_region_size', 'log_exact_region_size']
    if metric in size_metrics:
        def relative_length(x):
            x_average_run = x.groupby('name', dropna=False, observed=True)['value'].mean()
            min_average, max_average = x_average_run.min(), x_average_run.max()

            if min_average == max_average:
                return x['value']  # Avoid division by zero by returning original values

            return (x['value'] - min_average) / (max_average - min_average)

        df['value'] = df.groupby(['dataset'], dropna=False, observed=True).apply(relative_length).reset_index(level=0, drop=True)

        #df['value'] = np.log(df['value'].astype(np.float32))
    
    if plot_type == 'pointplot':
        g = sns.pointplot(
            data=df,
            x="dataset", 
            y="value",
            hue="name",
            #markers=["o", "v", "^", "s"],
            linestyles=[style for name, style in style_map.items() if name in names],
            markersize=3,
            errorbar=None if metric in size_metrics else ('se', 1),
            #palette=cmap,
            lw=1.5,
            ax=ax,
            palette=color_map,
        )
    elif plot_type == 'boxplot':
        g = sns.boxplot(
            data=df,
            x="dataset", 
            y="value",
            hue="name",
            palette=cmap,
            fliersize=2,
            ax=ax,
        )
    else:
        raise ValueError(f'Unknown plot type: {plot_type}')
    ax.set_xlabel(None)
    ax.set_ylabel(get_metric_name(metric))
    if metric in ['cond_cov_x_error', 'cond_cov_z_error', 'exact_region_size', 'score_time', 'total_time']:
        ax.set_yscale('log')
    if metric in ['coverage', 'wsc']:
        ax.axhline(1 - config.alpha, ls='--', color='black', lw=1)
        if metric == 'coverage':
            ax.set_ylim(0.7, 0.9)
        elif metric == 'wsc':
            ax.set_ylim(0.45, 0.9)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return g


def plot_comparison_multiple(df, config, plot_type='pointplot', cmap=None, ncols=3, metrics=None):
    #metrics = ['cond_cov_x_error', 'cond_cov_z_error', 'wsc']
    if metrics is None:
        metrics = ['cond_cov_x_error', 'cond_cov_z_error', 'wsc', 'coverage', 'median_region_size', 'total_time']
    n_datasets = df['dataset'].nunique()
    base_width = 0.3 if plot_type == 'pointplot' else 0.5
    nrows = np.ceil(len(metrics) / ncols).astype(int)
    fig, ax = plt.subplots(
        nrows, ncols, figsize=(base_width * n_datasets * ncols, nrows * 2.5), sharex=True
    )
    ax = ax.flatten()
    for axis, metric in zip(ax, metrics):
        plot_df = df.query('metric == @metric')
        plot_df = plot_df.reset_index()
        if plot_df['value'].isna().all():
            raise ValueError(f'All values are NaN for metric {metric}')
        g = plot_comparison(plot_df, config, axis, metric, plot_type=plot_type, cmap=cmap)
        if metric != 'test_coverage_time':
            handles, labels = axis.get_legend_handles_labels()
        g.get_legend().remove()
    
    bbox_y = 1.1 if nrows == 1 else 1.04
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, bbox_y), ncol=10, frameon=False)
    fig.tight_layout()



def plot_comparison_single(df, config, path, plot_type='pointplot', cmap=None):
    for metric in main_metrics:
        plot_df = df.query('metric == @metric')
        plot_df = plot_df.reset_index()
        if plot_df['value'].isna().all():
            continue
        base_width = 0.3 if plot_type == 'pointplot' else 0.5
        n_datasets = df['dataset'].nunique()
        fig, ax = plt.subplots(figsize=(base_width * n_datasets, 2.5))
        g = plot_comparison(plot_df, config, ax, metric, plot_type=plot_type, cmap=cmap)
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.24), ncol=3, frameon=False)
        g.get_legend().remove()
        fig.tight_layout()
        savefig(path / f'{metric}.pdf')

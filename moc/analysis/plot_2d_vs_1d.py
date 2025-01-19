from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from moc.conformal.conformalizers_manager import conformalizers
from moc.utils.general import savefig
from moc.metrics.cache import Cache, EmptyCache
from .helpers import create_name_from_dict


def plot_2D_contour_vs_1D_at_coverage(axis, x_value, conformalizer, alpha, ylim, zlim, grid_side=50, cache={}, **kwargs):
    device = conformalizer.model.device
    y1, y2 = torch.linspace(*ylim, grid_side, device=device), torch.linspace(*zlim, grid_side, device=device)
    Y1, Y2 = torch.meshgrid(y1, y2, indexing='ij')
    pos = torch.dstack((Y1, Y2))
    pos = pos[:, :, None, :]
    assert pos.shape == (y1.shape[0], y1.shape[0], 1, 2)
    x_pos = torch.tensor([[x_value.item()]], device=device)
    mask = conformalizer.is_in_region(x_pos, pos, alpha, cache=cache)
    mask = mask[:, :, 0]

    Y1, Y2, mask = Y1.cpu().numpy(), Y2.cpu().numpy(), mask.float().cpu().numpy()
    assert Y1.shape == Y2.shape == mask.shape

    fig2D, ax2D = plt.subplots()
    contour = ax2D.contour(Y1, Y2, mask, levels=[0], colors='r')
    plt.close(fig2D)
    
    contour_paths = contour.get_paths()
    if len(contour_paths) > 0:
        contour_path = contour_paths[0]
        for contour_points in contour_path.to_polygons():
            axis.plot(
                np.full_like(contour_points[:, 0], x_value.item()), 
                contour_points[:, 0],  # y-coordinates
                contour_points[:, 1],  # z-coordinates
                zorder=5,
                **kwargs,
            )


def plot_2D_region_vs_1D_on_axis(ax, hparams, datamodule, model, x_test, cache_calib=None, cache_test=None, **kwargs):
    data = datamodule.calib_dataloader()
    hparams = hparams.copy()
    method = hparams.pop('posthoc_method')

    levels = [0.2, 0.4, 0.8]
    colors = ('black', 'tab:green', '#FFD700')
    conformalizer = conformalizers[method](data, model, **hparams, cache_calib=cache_calib)
    if cache_test is None:
        cache_test = EmptyCache(datamodule.test_dataloader())
    for x_value, cache in zip(x_test, cache_test):
        for level, c in zip(levels, colors):
            plot_2D_contour_vs_1D_at_coverage(ax, x_value, conformalizer, 1 - level, color=c, cache=cache, **kwargs)
    ax.view_init(elev=18, azim=-25)


def scatter_2D_vs_1D(ax, x, y):
    ax.scatter(x[:, 0], y[:, 0], y[:, 1], alpha=0.4)


def plot_2D_region_vs_1D_per_method(hparams_list, datamodule, config, oracle_model=None, mqf2_model=None, path=None, grid_side=50, nrows=1):
    assert oracle_model is not None or mqf2_model is not None
    device = oracle_model.device if oracle_model is not None else mqf2_model.device

    size = len(hparams_list)
    if oracle_model is not None:
        size += 1
    ncols = np.ceil(size / nrows).astype(int)
    fig, ax = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.7 * nrows), subplot_kw={'projection': '3d'}, squeeze=False)
    ax = ax.flatten()
    for i in range(size, len(ax)):
        ax[i].set_visible(False)

    # Plot the calibration data and deduce the limits of the plot
    x, y = datamodule.data_calib[:]
    xlim = x[:, 0].min(), x[:, 0].max()
    ylim = y[:, 0].min(), y[:, 0].max()
    zlim = y[:, 1].min(), y[:, 1].max()

    # Create the x values for which we want to plot the regions
    eps = 1e-3
    x_test = torch.linspace(x.min() + eps, x.max() - eps, 10)
    #x_test = torch.tensor([(x.min() + x.max()) / 2])
    x_test = x_test.to(device)
    x_test = x_test[:, None]

    index = 0
    if oracle_model is not None:
        print(f'Plotting Oracle')
        scatter_2D_vs_1D(ax[index], x, y)
        plot_2D_region_vs_1D_on_axis(
            ax[index], 
            {'posthoc_method': 'HDR-H', 'n_samples': 500}, 
            datamodule, 
            oracle_model, 
            x_test, 
            ylim=ylim, 
            zlim=zlim, 
            grid_side=grid_side
        )
        ax[index].set_title('Oracle', y=0.97)
        index += 1
    if mqf2_model is not None:
        # Create the cache, ensuring faster computation and the same samples for all methods and alphas
        cache_calib = Cache(mqf2_model, datamodule.calib_dataloader(), n_samples=100, add_second_sample=True)
        y_test = torch.full(x_test.shape, torch.nan, device=device)
        dataset = TensorDataset(x_test, y_test)
        test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        cache_test = Cache(mqf2_model, test_dataloader, n_samples=100, add_second_sample=True)
        for hparams in hparams_list:
            method = hparams['posthoc_method']
            print(f'Plotting {method}')
            axis = ax[index]
            #hparams = {'posthoc_method': method}
            plot_2D_region_vs_1D_on_axis(
                axis, 
                hparams, 
                datamodule, 
                mqf2_model, 
                x_test, 
                ylim=ylim, 
                zlim=zlim, 
                grid_side=grid_side, 
                cache_calib=cache_calib, 
                cache_test=cache_test
            )
            axis.set_title(create_name_from_dict(hparams, config), y=0.97)
            index += 1
    
    for i, axis in enumerate(ax):
        axis.set(xlim=xlim, ylim=ylim, zlim=zlim)
        axis.set_xlabel('$X$', labelpad=-1)
        axis.set_ylabel('$Y_1$', labelpad=-1)
        axis.set_zlabel('$Y_2$', labelpad=-4)
        axis.xaxis.set_tick_params(pad=0)
        axis.yaxis.set_tick_params(pad=0)
        axis.zaxis.set_tick_params(pad=0)
    fig.text(0.93, 0.5, ' ', transform=fig.transFigure)
    fig.subplots_adjust(hspace=0.05, wspace=0.2)
    if path is not None:
        savefig(path)


def plot_2D_region_vs_1D_per_model(datamodule, config, oracle_model, trained_models, path=None, grid_side=50):
    method = 'HDR-CP'
    fig, ax = plt.subplots(1, len(trained_models) + 1, figsize=(20, 5), subplot_kw={'projection': '3d'})

    # Deduce the limits of the plot
    x, y = datamodule.data_calib[:]
    xlim = x[:, 0].min(), x[:, 0].max()
    ylim = y[:, 0].min(), y[:, 0].max()
    zlim = y[:, 1].min(), y[:, 1].max()

    # Create the x values for which we want to plot the regions
    eps = 1e-3
    x_test = torch.linspace(x.min() + eps, x.max() - eps, 10)
    x_test = x_test[:, None]

    print(f'Plotting Oracle')
    plot_2D_region_vs_1D_on_axis(
        ax[0], 
        {'posthoc_method': 'HDR-H', 'n_samples': 500}, 
        datamodule, 
        oracle_model, 
        x_test,
        ylim=ylim, 
        zlim=zlim, 
        grid_side=grid_side
    )
    for i, (model_name, model) in enumerate(trained_models.items()):
        print(f'Plotting {model_name}')
        axis = ax[i + 1]
        plot_2D_region_vs_1D_on_axis(
            axis, 
            {'posthoc_method': method}, 
            datamodule, 
            model, 
            x_test,
            ylim=ylim, 
            zlim=zlim, 
            grid_side=grid_side
        )
        axis.set_title(model_name, y=0.97)
    
    for i, axis in enumerate(ax):
        axis.set(xlim=xlim, ylim=ylim, zlim=zlim)
        axis.set(xlabel='$X$', ylabel='$Y_1$')
        if i == len(trained_models):
            axis.set(zlabel='$Y_2$')
    if path is not None:
        savefig(path)


def plot_2D_region_vs_1D_for_HDR_CP_per_n_samples(datamodule, config, model, path=None, grid_side=50):
    method = 'HDR-CP'
    sample_sizes = [5, 10, 30, 100, 300]
    fig, ax = plt.subplots(1, len(sample_sizes), figsize=(20, 5), subplot_kw={'projection': '3d'})

    # Deduce the limits of the plot
    x, y = datamodule.data_calib[:]
    xlim = x[:, 0].min(), x[:, 0].max()
    ylim = y[:, 0].min(), y[:, 0].max()
    zlim = y[:, 1].min(), y[:, 1].max()

    # Create the x values for which we want to plot the regions
    eps = 1e-3
    x_test = torch.linspace(x.min() + eps, x.max() - eps, 10)
    x_test = x_test[:, None]

    for i, sample_size in enumerate(sample_sizes):
        print(f'Plotting with {sample_size} samples')
        axis = ax[i]
        plot_2D_region_vs_1D_on_axis(
            axis, 
            {'posthoc_method': method, 'n_samples': sample_size}, 
            datamodule, 
            model, 
            x_test,
            ylim=ylim, 
            zlim=zlim, 
            grid_side=grid_side
        )
        axis.set_title(f'{sample_size} samples', y=0.97)
    
    for i, axis in enumerate(ax):
        axis.set(xlim=xlim, ylim=ylim, zlim=zlim)
        axis.set(xlabel='$X$', ylabel='$Y_1$')
        if i == len(sample_sizes) - 1:
            axis.set(zlabel='$Y_2$')
    if path is not None:
        savefig(path)

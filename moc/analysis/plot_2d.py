from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from moc.conformal.conformalizers_manager import conformalizers
from moc.utils.general import savefig
from moc.metrics.cache import Cache, EmptyCache


def plot_contour_at_coverage_2D(axis, x_value, conformalizer, method_name, alpha, ylim, zlim, color, grid_side=50, cache={}, **kwargs):
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

    axis.contourf(Y1, Y2, np.logical_not(mask), levels=[0, 0.0001], colors=[color], alpha=0)

    fig2D, ax2D = plt.subplots()
    contour = ax2D.contour(Y1, Y2, mask, levels=[0], colors=[color])
    plt.close(fig2D)
    
    if hasattr(contour, 'collections'):
        contour_paths = contour.collections[0].get_paths()
    else:
        contour_paths = contour.get_paths()
    if len(contour_paths) > 0:
        contour_path = contour_paths[0]
        for i, contour_points in enumerate(contour_path.to_polygons()):
            axis.plot(
                contour_points[:, 0],  
                contour_points[:, 1],
                label=method_name if i == 0 else None,
                color=color,
                **kwargs,
            )


def plot_2D_region(ax, tau, method_name, hparams, datamodule, model, x_test, color, cache_calib=None, cache_test=None, **kwargs):
    data = datamodule.calib_dataloader()
    method = hparams.pop('method')

    conformalizer = conformalizers[method](data, model, **hparams, cache_calib=cache_calib)
    if cache_test is None:
        cache_test = EmptyCache(datamodule.test_dataloader())
    for x_value, cache in zip(x_test, cache_test):
        plot_contour_at_coverage_2D(ax, x_value, conformalizer, method_name, 1 - tau, cache=cache, color=color, **kwargs)


def plot_2D_region_per_method(ax, x_value, tau, datamodule, oracle_model, mqf2_model, grid_side=50, custom_xlim=None, custom_ylim=None):
    methods = ['PCP', 'C-PCP', 'DR-CP', 'C-HDR', 'M-CP', 'L-CP']
    palette = sns.color_palette('Paired')
    style_dict = {
        'Oracle': ('black', '-'),
        'M-CP': (palette[5], '-'),
        'L-CP': ('tab:orange', '-'),
        'DR-CP': (palette[1], '-'),
        'C-HDR': (palette[3], '-'),
        'PCP': (palette[0], '--'),
        'C-PCP': (palette[2], '--'),
    }
    
    device = mqf2_model.device

    # Deduce the limits of the plot
    _, y = datamodule.data_train[:]
    xlim = y[:, 0].min(), y[:, 0].max()
    ylim = y[:, 1].min(), y[:, 1].max()

    x_train, y_train = datamodule.data_train[:]
    y_near = y_train[(x_train[:, 0] > x_value - 0.02) & (x_train[:, 0] < x_value + 0.02)]
    ax.scatter(y_near[:, 0], y_near[:, 1], color='gray', edgecolors='none', s=10, alpha=0.6)

    # Create the x value for which we want to plot the regions
    x_test = torch.as_tensor([[x_value]], dtype=torch.float32).to(device)

    # Create the cache, ensuring faster computation and the same samples for all methods and alphas
    cache_calib = Cache(mqf2_model, datamodule.calib_dataloader(), n_samples=100, add_second_sample=True)
    y_test = torch.full(x_test.shape, torch.nan, device=device)
    dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    cache_test = Cache(mqf2_model, test_dataloader, n_samples=100, add_second_sample=True)

    if oracle_model is not None:
        color, linestyle = style_dict['Oracle']
        plot_2D_region(
            ax,
            tau,
            'Oracle',
            {'method': 'HDR-H', 'n_samples': 500}, 
            datamodule, 
            oracle_model, 
            x_test,
            ylim=xlim, 
            zlim=ylim, 
            grid_side=grid_side,
            color=color,
            linestyle=linestyle
        )
    for i, method in enumerate(methods):
        color, linestyle = style_dict[method]
        plot_2D_region(
            ax,
            tau,
            method,
            {'method': method}, 
            datamodule, 
            mqf2_model, 
            x_test, 
            ylim=xlim, 
            zlim=ylim, 
            grid_side=grid_side, 
            cache_calib=cache_calib, 
            cache_test=cache_test,
            color=color,
            linestyle=linestyle
        )

        if x_value == 0 and tau == 0.2:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=6, frameon=False)
    
    if custom_xlim is None and custom_ylim is None:
        ax.set(xlim=xlim, ylim=ylim)
    else:
        ax.set(xlim=custom_xlim, ylim=custom_ylim)
    ax.set(xlabel='$Y_1$', ylabel='$Y_2$')

    if tau == 0.2:
        ax.set_title(f'X = {x_value}', pad=10)
    if x_value == -1:
        ax.text(-0.3, 0.5, r'$\tau = $' + str(tau), rotation=90, va='center', ha='center', transform=ax.transAxes)
    

def plot_2D_regions_by_x_and_tau(datamodule, oracle_model, mqf2_model, path=None, grid_side=50):
    x_values = [-1, 0, 1]
    tau_values = [0.2, 0.4, 0.8]

    fig, axs = plt.subplots(len(tau_values), len(x_values), figsize=(12, 12))

    for x_step, x_value in enumerate(tqdm(x_values)):
        for y_step, tau_value in enumerate(tau_values):
            plot_2D_region_per_method(axs[y_step, x_step], x_value, tau_value, datamodule, oracle_model, mqf2_model, grid_side)

    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    if path is not None:
        savefig(path)

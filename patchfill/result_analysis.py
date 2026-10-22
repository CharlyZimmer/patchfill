import xarray as xr
import dask.array as da
import numpy as np
import tqdm as tqdm
import random
import h5py
import torch
import matplotlib.pyplot as plt
from einops import rearrange

from patchfill.preprocessing import reverse_normalization



def measurements_preserved(path, var):

    ds = xr.open_zarr(path + 'no_gaps.zarr')
    
    for i in tqdm.tqdm(ds['time']):
        df = ds.sel({'time' : i.data}, method='nearest').to_dataframe()
        mask = df[var].notna()
    
        # get all measurements that are on land surface
        filtered_df = df.loc[mask, [var, var + '_no_gaps']]
        filtered_df = filtered_df.loc[df['land_mask'], [var, var + '_no_gaps']]
        
        comparison = (filtered_df[var] == filtered_df[var + '_no_gaps']).all()
        if not comparison:
            first_false_index = np.argmax(~(filtered_df[var] == filtered_df[var + '_no_gaps'])).item()
            print(first_false_index)
            print(filtered_df[var].shape)
            print(filtered_df[var].iloc[first_false_index])
            print(filtered_df[var + '_no_gaps'].iloc[first_false_index])
            return False

    return True

def all_gaps_filled(path, var):

    ds = xr.open_zarr(path + 'no_gaps.zarr')
    
    for i in tqdm.tqdm(ds['time']):
        df = ds.sel({'time' : i.data}, method='nearest').to_dataframe()
        mask = df[var].isna()
    
        # get all NaNs that are on the land surface
        filtered_df = df.loc[mask, [var + '_no_gaps']]
        filtered_df = filtered_df.loc[df['land_mask'], [var + '_no_gaps']]
        
        comparison = (filtered_df[var + '_no_gaps'].notna()).all()
        if not comparison:
            return False

    return True

def get_datacube_dist(path):
    # measurement distribution
    with open(path + 'data_cube_dist.npy', 'rb') as f:
        h_measurements = np.load(f)
        bins_measurements = np.load(f)
    return h_measurements, bins_measurements

def get_pred_dist(path, var):
    
    ds = xr.open_zarr(path + 'no_gaps.zarr')
    
    # distribution in gap-filled cube
    h, bins =da.histogram(ds[var + '_no_gaps'], bins=100, range=[150, 350])
    h = h.compute()
    only_predictions = ds[var + '_no_gaps'].where(ds[var].isnull())

    return only_predictions


def pred_dist(path, var, bins, val_range):

    h_measurements, bins_measurements = get_datacube_dist(path)
    plt.stairs(h_measurements, bins_measurements, color='b', label='Measurements')
    
    only_predictions = get_pred_dist(path, var)
    
    h, bins =da.histogram(only_predictions, bins=bins, range=val_range)
    h = h.compute()
    
    max_value = h.max()
    h = h / max_value
    
    plt.stairs(h, bins, color='r', label='Predictions')
    
    plt.title('Distribution of Predicted Values')
    plt.xlabel(var)
    plt.ylabel('Normalized Amount')
    plt.legend()

def estimation_comparison(path, var, bins, val_range):
    
    with open(path + 'estimation_dist.npy', 'rb') as f:
        h_estimation = np.load(f)
        bins_estimation = np.load(f)
    plt.stairs(h_estimation, bins_estimation, color='b', label='Temporal Mean Estimation')

    only_predictions = get_pred_dist(path, var)
    
    h, bins =da.histogram(only_predictions, bins=bins, range=val_range)
    h = h.compute()
    
    max_value = h.max()
    h = h / max_value
    
    plt.stairs(h, bins, color='r', label='Predictions')
    
    plt.title('LST Distribution')
    plt.xlabel('Land Surface Temperature (K)')
    plt.ylabel('Normalized Amount')
    plt.legend()

def plot_patch_predictions(path, model, cmap, val_range, cmap_range, block_size, patch_overlap, idx=None):
    fig, axs = plt.subplots(10,6, figsize=(30, 50))
    # cmap.set_under('w')
    
    vmin = cmap_range[0]
    vmax = cmap_range[1]
    fontsize = 15
    
    with h5py.File(path + 'dataset.h5', 'r') as f:

        num_test_samples = f['X_test'].shape[0]

        samples = []
        targets = []
        indices = []

        num_random_samples = 5
        if idx:
            num_random_samples = 4
            samples.append(f['X_test'][idx])
            targets.append(f['y_test'][idx])
            indices.append(idx)
        
        for i in random.sample(range(0, num_test_samples), num_random_samples):
            samples.append(f['X_test'][i])
            targets.append(f['y_test'][i])
            indices.append(i)

        samples = np.array(samples)
        targets = np.array(targets)
        predictions = model.predict(samples)

        samples = reverse_normalization(samples, val_range)
        targets = reverse_normalization(targets, val_range)
        predictions = reverse_normalization(predictions, val_range)

        # only show 5 patches
        # cut = int((samples.shape[1] - 5)/2)
        # samples = samples[:,cut:-cut,...]
        # print(samples.shape)
        
        for key, value in zip(range(0,10,2), range(5)):
    
            X = samples[value]
            y = targets[value]
            pred = predictions[value]
            
            axs[key, 0].matshow(np.squeeze(X[0, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key, 1].matshow(np.squeeze(X[1, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key, 2].matshow(np.squeeze(X[2, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key, 3].matshow(np.squeeze(X[3, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key, 4].matshow(np.squeeze(X[4, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key, 5].matshow(np.zeros((block_size[1],block_size[2])), cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key, 5].set_title(str(indices[value]), fontsize=fontsize)
    
            axs[key+1, 0].matshow(np.zeros((block_size[1],block_size[2])), cmap=cmap, vmin=vmin, vmax=vmax)
            mask = ~np.isnan(y)
            loss = np.mean(np.abs(y[mask] - pred[mask]))
            axs[key+1, 0].set_title(str(loss), fontsize=fontsize)
            
            mask = np.zeros((block_size[1],block_size[2]))
            mask[np.isnan(np.squeeze(X[2, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0]))] = vmin
            mask[np.isnan(np.squeeze(y))] = 0
            axs[key+1, 1].matshow(mask, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key+1, 1].set_title('mask', fontsize=fontsize)
            
            y[np.isnan(y)] = X[2, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0][np.isnan(np.squeeze(y))]
            helper = np.full((X.shape[1], X.shape[2]), np.nan)
            helper[patch_overlap:-patch_overlap, patch_overlap:-patch_overlap] = np.squeeze(y)
            axs[key+1, 2].matshow(helper, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key+1, 2].set_title('target', fontsize=fontsize)

            helper[patch_overlap:-patch_overlap, patch_overlap:-patch_overlap] = np.squeeze(pred)
            axs[key+1, 4].matshow(helper, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key+1, 4].set_title('pred', fontsize=fontsize)
            
            # insert known values into pred
            pred = np.squeeze(pred)
            pred[~np.isnan(X[2, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0])] = X[2, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0][~np.isnan(X[2, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0])]
            helper[patch_overlap:-patch_overlap, patch_overlap:-patch_overlap] = np.squeeze(pred)
            axs[key+1, 3].matshow(helper, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key+1, 3].set_title('result', fontsize=fontsize)

            previous_gap = np.copy(np.squeeze(y))
            previous_gap[np.isnan(np.squeeze(X[1, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0]))] = np.nan
            axs[key+1, 5].matshow(previous_gap, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key+1, 5].set_title('previous gap', fontsize=fontsize)
        
    plt.subplots_adjust(hspace=0.4)
    
    for i in range(len(axs)):
        for j in range(len(axs[0])):        
            axs[i][j].set_yticklabels([])
            axs[i][j].set_xticklabels([])
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])

def plot_patch_predictions_pytorch(X_test, y_test, model, cmap, val_range, cmap_range, block_size, patch_overlap, idx=None):
    fig, axs = plt.subplots(10,6, figsize=(30, 50))
    # cmap.set_under('w')
    
    vmin = cmap_range[0]
    vmax = cmap_range[1]
    fontsize = 15
    
    num_test_samples = X_test.shape[0]

    samples = []
    targets = []
    indices = []

    num_random_samples = 5
    if idx:
        num_random_samples = 4
        samples.append(X_test[idx])
        targets.append(y_test[idx])
        indices.append(idx)
    
    for i in random.sample(range(0, num_test_samples), num_random_samples):
        samples.append(X_test[i])
        targets.append(y_test[i])
        indices.append(i)

    samples = torch.stack(samples, dim=0)
    targets = torch.stack(targets, dim=0)
    
    predictions, patch_mask = model(samples)

    predictions = rearrange(predictions, 'b c h w -> b h w c')
    predictions = predictions.unsqueeze(1)

    targets = rearrange(targets, 'b c h w -> b h w c')
    targets = targets.unsqueeze(1)

    samples = rearrange(samples, 'b t c h w -> b t h w c')

    predictions = predictions.detach().cpu().numpy()
    targets = targets.cpu().numpy()
    samples = samples.cpu().numpy()

    samples = reverse_normalization(samples, val_range)
    targets = reverse_normalization(targets, val_range)
    predictions = reverse_normalization(predictions, val_range)

    # only show 5 patches
    # cut = int((samples.shape[1] - 5)/2)
    # samples = samples[:,cut:-cut,...]
    # print(samples.shape)
    
    for key, value in zip(range(0,10,2), range(5)):

        X = samples[value]
        y = targets[value]
        pred = predictions[value]
        
        axs[key, 0].matshow(np.squeeze(X[0, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
        axs[key, 1].matshow(np.squeeze(X[1, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
        axs[key, 2].matshow(np.squeeze(X[2, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
        axs[key, 3].matshow(np.squeeze(X[3, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
        axs[key, 4].matshow(np.squeeze(X[4, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
        axs[key, 5].matshow(np.zeros((block_size[1],block_size[2])), cmap=cmap, vmin=vmin, vmax=vmax)
        axs[key, 5].set_title(str(indices[value]), fontsize=fontsize)

        axs[key+1, 0].matshow(np.zeros((block_size[1],block_size[2])), cmap=cmap, vmin=vmin, vmax=vmax)
        mask = ~np.isnan(y)
        loss = np.mean(np.abs(y[mask] - pred[mask]))
        axs[key+1, 0].set_title(str(loss), fontsize=fontsize)
        
        mask = np.zeros((block_size[1],block_size[2]))
        mask[np.isnan(np.squeeze(X[2, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0]))] = vmin
        mask[np.isnan(np.squeeze(y))] = 0
        axs[key+1, 1].matshow(mask, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[key+1, 1].set_title('mask', fontsize=fontsize)

        # if patch_overlap == 0:
        #     y[np.isnan(y)] = X[2, :, :, 0][np.isnan(np.squeeze(y))]    
        # else:
        #     y[np.isnan(y)] = X[2, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0][np.isnan(np.squeeze(y))]
            
        helper = np.full((X.shape[1], X.shape[2]), np.nan)

        if patch_overlap == 0:
            helper = np.squeeze(y)
        else:
            helper[patch_overlap:-patch_overlap, patch_overlap:-patch_overlap] = np.squeeze(y)
        
        axs[key+1, 2].matshow(helper, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[key+1, 2].set_title('target', fontsize=fontsize)

        if patch_overlap == 0:
            helper = np.squeeze(pred)
        else:
            helper[patch_overlap:-patch_overlap, patch_overlap:-patch_overlap] = np.squeeze(pred)
        
        axs[key+1, 4].matshow(helper, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[key+1, 4].set_title('pred', fontsize=fontsize)
        
        # insert known values into pred
        pred = np.squeeze(pred)

        if patch_overlap == 0: 
            pred[~np.isnan(X[2, :, :, 0])] = X[2, :, :, 0][~np.isnan(X[2, :, :, 0])]
            helper = np.squeeze(pred)
        else:
            pred[~np.isnan(X[2, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0])] = X[2, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0][~np.isnan(X[2, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0])]
            helper[patch_overlap:-patch_overlap, patch_overlap:-patch_overlap] = np.squeeze(pred)
            
        axs[key+1, 3].matshow(helper, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[key+1, 3].set_title('result', fontsize=fontsize)

        previous_gap = np.copy(np.squeeze(y))

        if patch_overlap == 0:
            previous_gap[np.isnan(np.squeeze(X[1, :, :, 0]))] = np.nan    
        else:
            previous_gap[np.isnan(np.squeeze(X[1, patch_overlap:-patch_overlap, patch_overlap:-patch_overlap, 0]))] = np.nan
        
        axs[key+1, 5].matshow(previous_gap, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[key+1, 5].set_title('previous gap', fontsize=fontsize)
    
    plt.subplots_adjust(hspace=0.4)
    
    for i in range(len(axs)):
        for j in range(len(axs[0])):        
            axs[i][j].set_yticklabels([])
            axs[i][j].set_xticklabels([])
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])

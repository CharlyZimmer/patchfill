import h5py
import matplotlib.pyplot as plt
import random
import numpy as np
import xarray as xr
import dask.array as da
import tqdm
from global_land_mask import globe
from patchfill.preprocessing import iter_data_var_blocks, calculate_total_chunks, get_chunk_sizes
from global_land_mask import globe
import geopandas as gpd
import cmocean

from patchfill.preprocessing import reverse_normalization

_world_gdf = None

def _get_world():
    global _world_gdf
    if _world_gdf is None:
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        _world_gdf = gpd.read_file(url)
    return _world_gdf

def get_dataset_size(path):
    with h5py.File(path + 'dataset.h5', 'r') as f:
        X_train = f['X_train']
        y_train = f['y_train']
        X_test = f['X_test']
        y_test = f['y_test']
        time_train = f['time_train']
        time_test = f['time_test']
        lat_train = f['lat_train']
        lat_test = f['lat_test']
        lon_train = f['lon_train']
        lon_test = f['lon_test']
        
        print("X train shape:", X_train.shape)
        print("y train shape:", y_train.shape)
        print("X test shape:", X_test.shape)
        print("y test shape:", y_test.shape)
        print("time train shape:", time_train.shape)
        print("time test shape:", time_test.shape)

def plot_random_samples(path, cmap, cmap_range, val_range):
    with h5py.File(path + 'dataset.h5', 'r') as f:
        
        fig, axs = plt.subplots(10,6, figsize=(30, 50))
    
        # cmap.set_under('w')

        vmin = cmap_range[0]
        vmax = cmap_range[1]

        X_train = f['X_train']
        middle_index = int((X_train.shape[1]-1)*0.5)
        for key, value in zip(range(0,10,1), random.sample(range(0, X_train.shape[0]), 10)):
            X = f['X_train'][value]
            y = f['y_train'][value]
            time = f['time_train'][value]
            lat = f['lat_train'][value]
            lon = f['lon_train'][value]
            X = reverse_normalization(X, val_range)
            y = reverse_normalization(y, val_range)
            axs[key, 0].matshow(np.squeeze(X[middle_index-2, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key, 1].matshow(np.squeeze(X[middle_index-1, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key, 5].matshow(np.squeeze(X[middle_index, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key, 3].matshow(np.squeeze(X[middle_index+1, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)
            axs[key, 4].matshow(np.squeeze(X[middle_index+2, ..., 0]), cmap=cmap, vmin=vmin, vmax=vmax)

            y[np.isnan(y)] = X[2, ..., 0][np.isnan(np.squeeze(y))]
            axs[key, 2].matshow(np.squeeze(y), cmap=cmap, vmin=vmin, vmax=vmax)
    
            # label = ('time start: ' + str(time[0][:10]) 
            #         + '\n time stop: ' + str(time[-1][:10])
            #         + '\n lat start: ' + str(lat[0])
            #         + '\n lat stop: ' + str(lat[-1])
            #         + '\n lon start: ' + str(lon[0])
            #         + '\n lon stop: ' + str(lon[-1])
            #         + '\n id: ' + str(value))

            label = ''
            
            bbox = axs[key, 0].get_yticklabels()[-1].get_window_extent()
            x,_ = axs[key, 0].transAxes.inverted().transform([bbox.x0, bbox.y0])
            axs[key, 0].set_title(label, ha='left', x=x)
        
        plt.subplots_adjust(hspace=0.4)
    
        for i in range(len(axs)):
            for j in range(len(axs[0])):        
                axs[i][j].set_yticklabels([])
                axs[i][j].set_xticklabels([])
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])

def empty_targets(path):
    with h5py.File(path + 'dataset.h5', 'r') as f:
        for i in tqdm.tqdm(range(0, f['y_train'].shape[0], 2000)):
            batch = np.squeeze(f['y_train'][i:i+2000])
    
            nan_patches_count = np.sum(np.isnan(batch).all(axis=(1, 2)))
    
            if nan_patches_count != 0:
                return True

    return False

def compute_datacube_dist(ds, var, bins, val_range, path):
    h_datacube, bins = da.histogram(ds[var], bins=300, range=val_range)
    h_datacube = h_datacube.compute()
    
    max_value = h_datacube.max()
    h_datacube = h_datacube / max_value
    
    with open(path + 'data_cube_dist.npy', 'wb') as f:
        np.save(f, h_datacube)
        np.save(f, bins)
    
def compute_train_dataset_dist(path, bins, val_range):
    
    h_train = np.zeros(len(bins)-1)
    with h5py.File(path + 'dataset.h5', 'r') as f:
        num_train_samples = f['X_train'].shape[0]
        batch_size = 10000
        
        counter = 0
        for i in tqdm.tqdm(range(0, num_train_samples, batch_size)):
            batch = f['X_train'][counter:counter+batch_size]
            batch = reverse_normalization(batch, val_range)
            counter += batch_size
            
            batch_hist = np.histogram(batch, bins=bins)
            h_train+= batch_hist[0]
    
    max_value = h_train.max()
    h_train = h_train / max_value
            
    with open(path + 'train_dataset_dist.npy', 'wb') as f:
        np.save(f, h_train)
        np.save(f, bins)

def plot_dist_comparison(h_datacube, h_train, bins):
    plt.stairs(h_datacube, bins, color='b', label='Data Cube')
    plt.stairs(h_train, bins, color='r', label='Train Dataset')
    
    plt.title('Distribution Comparison')
    plt.xlabel('Land Surface Temperature (K)')
    plt.ylabel('Normalized Amount')
    plt.legend()

def plot_global_map(ds, var, cmap, val_range, time=None, title=None):
    """Plot a global map of `var` from an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with lat/lon dimensions (optionally a time dimension).
    var : str
        Variable to plot.
    cmap : colormap
    val_range : tuple
        (vmin, vmax) for the colormap.
    time : str, optional
        If given, select the nearest time slice; the actual selected timestamp
        is used as the plot title unless *title* is also provided.
    title : str, optional
        Explicit title override.
    """
    if time is not None:
        ds = ds.sel(time=time, method='nearest')
        actual_time = str(ds['time'].values)[:10]
        if title is None:
            title = actual_time

    da = ds[var]

    vmin, vmax = val_range
    vmid = (vmin + vmax) / 2

    world = _get_world()
    fig, ax = plt.subplots(figsize=(15, 10))
    world.plot(ax=ax, color='white', edgecolor='black')

    im = ax.pcolormesh(
        da['lon'].values, da['lat'].values, da.values,
        cmap=cmap, vmin=vmin, vmax=vmax, shading='auto',
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_ticks([vmin, vmid, vmax])

    if title:
        ax.set_title(title)
    ax.set_ylabel('Latitude [°]')
    ax.set_xlabel('Longitude [°]')
    plt.show()

def plot_regional_map(df, var, cmap, val_range, title, region_bounds=None):
    import matplotlib.pyplot as plt

    dfdn = df.reset_index()

    vmin, vmax = val_range
    vmid = (vmin + vmax) / 2

    # Pivot to 2D grid: rows=lat, cols=lon
    pivot = dfdn.pivot(index="lat", columns="lon", values=var)
    lons = pivot.columns.values
    lats = pivot.index.values
    data = pivot.values

    fig, ax = plt.subplots(figsize=(15, 10))

    world = _get_world()
    world.plot(ax=ax, facecolor="none", edgecolor="black")

    # plot the gridded data
    im = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

    # restrict to region if requested
    if region_bounds:
        minx, miny, maxx, maxy = region_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_ticks([vmin, vmid, vmax])

    ax.set_title(title)
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")

    plt.show()



def compute_gap_map(path, var):

    ds = xr.open_zarr(path + 'preprocessed.zarr')
    nan_array = ds[var].isnull()
    nan_count = nan_array.sum(dim='time')
    amount_time_steps = ds[var].shape[0]
    return (nan_count / amount_time_steps) * 100  # DataArray, % missing

def gap_map(path, var):
    gap_da = compute_gap_map(path, var)

    ds = xr.open_zarr(path + 'preprocessed.zarr')
    land_mask = ds['land_mask'].isel(time=0)
    gap_da = gap_da.where(land_mask)

    plot_global_map(gap_da.to_dataset(name=var), var, cmap='Purples', val_range=[0, 100],
                    title='Frequency of Missing Values [%]')

def compute_mean_map(path, var, val_range):

    ds = xr.open_zarr(path + 'preprocessed.zarr')
    ds = reverse_normalization(ds, val_range)
    return ds[var].mean(dim='time', skipna=True)  # DataArray

def mean_map(path, var, cmap, val_range):
    mean_da = compute_mean_map(path, var, val_range)

    ds = xr.open_zarr(path + 'preprocessed.zarr')
    land_mask = ds['land_mask'].isel(time=0)
    mean_da = mean_da.where(land_mask)

    plot_global_map(mean_da.to_dataset(name=var), var, cmap=cmap, val_range=val_range,
                    title='Mean Values over Time')

def compute_gap_value_estimation(var, path, val_range, bins):

    mean_values = compute_mean_map(path, var, val_range)
    gap_map = compute_gap_map(path, var)
    
    temperatures = mean_values[var].values
    frequencies = gap_map[var].values
    
    hist, bin_edges = np.histogram(temperatures, bins=bins, range=val_range, weights=frequencies)
    
    max_value = hist.max()
    hist = hist / max_value
    
    with open(path + 'estimation_dist.npy', 'wb') as f:
        np.save(f, hist)
        np.save(f, bin_edges)

def gap_value_estimation(path, xlabel):
    
    with open(path + 'estimation_dist.npy', 'rb') as f:
        hist = np.load(f)
        bin_edges = np.load(f)
    
        plt.stairs(hist, bin_edges, color='b', label='Estimation')
        plt.xlabel(xlabel)
        plt.ylabel('Normalized Amount of Gaps')
        plt.title('Estimated Distribution in Gap Areas')

def sample_dist_lat(path):
    with h5py.File(path + 'dataset.h5', 'r') as f:
        lat_train = f['lat_test'][:]
    
    plt.hist(lat_train.ravel(), bins=500)
    plt.xlabel('Latitude')
    plt.ylabel('Amount of Samples')
    plt.show()

def sample_dist_lon(path):
    with h5py.File(path + 'dataset.h5', 'r') as f:
        lon_train = f['lon_test'][:]
    
    plt.hist(lon_train.ravel(), bins=500) # only show starting coordinate
    plt.xlabel('Longitude')
    plt.ylabel('Amount of Samples')
    plt.show()

    




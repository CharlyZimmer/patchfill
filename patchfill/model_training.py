import numpy as np
import tensorflow as tf
# import tensorflow.keras as keras
import keras
# import tensorflow.keras.layers as L
import keras.layers as L
# import tensorflow.keras.backend as K
import keras.backend as K
# from tensorflow.keras.layers import Layer
from keras.layers import Layer
import matplotlib.pyplot as plt
import random
import h5py
import tqdm
import einops
import time
import tqdm
import cmocean

from patchfill.preprocessing import reverse_normalization

import h5py
import tensorflow as tf
import numpy as np
import os

import h5py
import tensorflow as tf
import numpy as np
import os

def create_tf_dataset(file_path, x_dataset_name, y_dataset_name, batch_size, indices):
    def gen():
        with h5py.File(file_path, 'r') as f:
            x_data = f[x_dataset_name]
            y_data = f[y_dataset_name]
            while True:
                for start in range(0, len(indices), batch_size):
                    end = min(start + batch_size, len(indices))
                    batch_indices = indices[start:end]
                    x_batch = x_data[batch_indices]
                    y_batch = y_data[batch_indices]
                    yield x_batch, y_batch

    with h5py.File(file_path, 'r') as f:
        block_size = f[x_dataset_name].shape[1:]
        block_size_y = f[y_dataset_name].shape[1:] # nesessary for patch overlap
            
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, block_size[0], block_size[1], block_size[2], block_size[3]), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1, block_size_y[1], block_size_y[2], 1), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

def hdf5_to_tf_dataset(path, batch_size=128, validation_split=0.1, portion=1.0):

    hdf5_file_path = path + 'dataset.h5'
    
    with h5py.File(hdf5_file_path, 'r') as f:
        num_train_samples = f['X_train'].shape[0]
        num_test_samples = f['X_test'].shape[0]
    
    # Calculate the number of validation samples
    num_val_samples = int(num_train_samples * validation_split)
    num_train_samples = num_train_samples - num_val_samples
    
    # Create indices for training and validation sets
    all_indices = np.arange(num_train_samples + num_val_samples)
    train_indices = all_indices[:int(num_train_samples*portion)]
    val_indices = all_indices[num_train_samples:int(num_train_samples + num_val_samples*portion)]
    
    # Create tf.data.Dataset objects for training and validation
    train_dataset = create_tf_dataset(hdf5_file_path, 'X_train', 'y_train', batch_size, train_indices)

    val_dataset = create_tf_dataset(hdf5_file_path, 'X_train', 'y_train', batch_size, val_indices)
    # with h5py.File(hdf5_file_path, 'r') as f:
    #     X_val = f['X_train'][val_indices]
    #     y_val = f['y_train'][val_indices]
    
    test_dataset = create_tf_dataset(hdf5_file_path, 'X_test', 'y_test', batch_size, np.arange(num_test_samples))

    return train_dataset, val_dataset, test_dataset

def cnn_3d(block_size, kernel_size, channels, val_range, learning_rate, global_mean):
    activation = 'relu'
        
    class ReplaceMissingWithMean(tf.keras.layers.Layer):
        def __init__(self):
            super(ReplaceMissingWithMean, self).__init__()
        
        def compute_mean(self, inputs, global_mean):
    
            non_nan_mask = ~tf.math.is_nan(inputs)
            
            # Compute mean along spatial dimensions for valid values
            masked_tensor = tf.where(non_nan_mask, inputs, 0.0)
            valid_sum = tf.reduce_sum(masked_tensor, axis=(1, 2, 3), keepdims=True)
            valid_count = tf.reduce_sum(tf.cast(non_nan_mask, tf.float32), axis=(1, 2, 3), keepdims=True)
            mean_non_nan = tf.where(valid_count > 0, valid_sum / valid_count, 0.0)
        
            # Generate noise with the same shape as inputs
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=0.05)
            
            # Scale the noise by a small factor to control the range of variation
            noise_scale = 0.1
            noise = noise * noise_scale
            
            # Add noise to the mean tensor, broadcasting it to match the shape of inputs
            mean_with_noise = mean_non_nan + noise
            
            # Replace NaN values in inputs with the noisy mean
            cleaned_tensor = tf.where(tf.math.is_nan(inputs), mean_with_noise, inputs)
            
            result = tf.where(tf.math.is_nan(cleaned_tensor), global_mean, cleaned_tensor)
            return result

        def call(self, inputs):
            updated_channels = []
            for i in range(channels):
                updated_channel = self.compute_mean(inputs[..., i], global_mean[i])
                updated_channels.append(tf.expand_dims(updated_channel, axis=-1))

            # Concatenate the updated indices back
            updated_inputs = tf.concat(updated_channels, axis=-1)
            
            return updated_inputs

    
    def select_index(x):
        middle_index = int((x.shape[1]-1)*0.5)
        return tf.expand_dims(x[:, middle_index, :, :, :], axis=1)

    # def truncate(x):
    #     return x[:, 3:-3, :, :, :]
    
    def create_model(learning_rate):
    
        inputs = L.Input(name="input_stack", shape=(block_size[0],block_size[1],block_size[2],channels), dtype=tf.float64)

        # truncated = L.Lambda(truncate, output_shape=(3, block_size[1], block_size[2], 1))(inputs)
    
        replaced = ReplaceMissingWithMean()(inputs)
        
        #encoder
        conv1 = L.Conv3D(filters=32, kernel_size=kernel_size, padding='same', name="conv1", activation=activation)(replaced)
        down1 = L.Conv3D(filters=32, kernel_size=kernel_size, padding='same', name="down1", strides=(1,2,2), activation=activation)(conv1)
        conv2 = L.Conv3D(filters=64, kernel_size=kernel_size, padding='same', name="conv2", activation=activation)(down1)
        down2 = L.Conv3D(filters=64, kernel_size=kernel_size, padding='same', name="down2", strides=(1,2,2), activation=activation)(conv2)
        conv3 = L.Conv3D(filters=128, kernel_size=kernel_size, padding='same', name="conv3", activation=activation)(down2)
        down3 = L.Conv3D(filters=128, kernel_size=kernel_size, padding='same', name="down3", strides=(1,2,2), activation=activation)(conv3)
        
        #decoder
        up1 = L.Conv3DTranspose(filters=128, strides = (1,2,2), kernel_size=kernel_size, padding='same', name="up1", activation=activation)(down3)
        x = L.Add()([conv3 ,up1])
        conv4 = L.Conv3D(filters=128, kernel_size=kernel_size, padding='same', name="conv4", activation=activation)(x)
    
        up2 = L.Conv3DTranspose(filters=64, strides = (1,2,2), kernel_size=kernel_size, padding='same', name="up2", activation=activation)(conv4)
        x = L.Add()([conv2 ,up2])
        conv5 = L.Conv3D(filters=64, kernel_size=kernel_size, padding='same', name="conv5", activation=activation)(x)
        
        up3 = L.Conv3DTranspose(filters=32, strides = (1,2,2), kernel_size=kernel_size, padding='same', name="up3", activation=activation)(conv5)
        
        x = L.Add()([conv1 ,up3])
        conv6 = L.Conv3D(filters=1, kernel_size=kernel_size, activation=activation, padding='same')(x)
    
        outputs = L.Lambda(select_index, output_shape=(1, block_size[1], block_size[2], 1), dtype=tf.float64)(conv6)
    
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
        def masked_mae(y_true, y_pred):
            mask_nan = ~tf.math.is_nan(y_true)
            
            # Apply the mask to both y_true and y_pred
            masked_true = tf.boolean_mask(y_true, mask_nan)
            masked_pred = tf.boolean_mask(y_pred, mask_nan)
    
            # Compute the Mean Absolute Error
            mae = tf.reduce_mean(tf.abs(masked_true - masked_pred))
    
            return mae
        
        def masked_mae_rescaled(y_true, y_pred):
            mask_nan = ~tf.math.is_nan(y_true)

            y_true = reverse_normalization(y_true, val_range)
            y_pred = reverse_normalization(y_pred, val_range)
            
            y_true = tf.where(mask_nan, y_true, y_true)
            y_pred = tf.where(mask_nan, y_pred, y_pred)
    
            # Apply the mask to both y_true and y_pred
            masked_true = tf.boolean_mask(y_true, mask_nan)
            masked_pred = tf.boolean_mask(y_pred, mask_nan)
    
            # Compute the Mean Absolute Error
            mae = tf.reduce_mean(tf.abs(masked_true - masked_pred))
    
            return mae
        
        model.compile(
            loss=masked_mae, 
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[masked_mae_rescaled]
        )
    
        return model
    
    model = create_model(learning_rate)
    model.summary(line_length = 110)
    return model


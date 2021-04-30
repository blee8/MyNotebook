#%%

import os

import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt


class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=None, val_df=None, test_df=None,
               label_columns=None):
    # Store the raw data.
    if train_df is not None:
        self.train_df = train_df
    if val_df is not None:
        self.val_df = val_df
    if test_df is not None:
        self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'
    ])


def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]

  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)
  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])
  return inputs, labels

WindowGenerator.split_window = split_window


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=False, #default
#      shuffle=True,
      batch_size=16,)
  ds = ds.map(self.split_window)
  return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)
#  return self.make_dataset(self.train_df_s)

@property
def val(self):
  return self.make_dataset(self.val_df)
#  return self.make_dataset(self.val_df_s)

@property
def test(self):
  return self.make_dataset(self.test_df)
#  return self.make_dataset(self.test_df_s)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example


#%%

class WinGen():
  def __init__(self, input_width, label_width, shift,
               train_df=None, val_df=None, test_df=None,
               label_columns=None):
    # Store the raw data.
    if train_df is not None:
        self.train_df = train_df
    if val_df is not None:
        self.val_df = val_df
    if test_df is not None:
        self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'
    ])


def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  lab = features[:, self.labels_slice, 0:2]

  #global lab
#  lab = tf.identity(labels)
#  lab = tf.stack([lab[:, :, 0:2]])
#  lab = lab[0,:,:,:]

  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)
  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

#  lab.set_shape([None, self.label_width, None])
  return inputs, labels, lab

WinGen.split_window = split_window


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=False, #default
#      shuffle=True,
      batch_size=16,)
  ds = ds.map(self.split_window)
  # ds = ds.map(lambda input, lable : x[:, self.input_slice, :], ......)

  return ds

WinGen.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df)
#  return self.make_dataset(self.train_df_s)

@property
def val(self):
  return self.make_dataset(self.val_df)
#  return self.make_dataset(self.val_df_s)

@property
def test(self):
  return self.make_dataset(self.test_df)
#  return self.make_dataset(self.test_df_s)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result



WinGen.train = train
WinGen.val = val
WinGen.test = test
WinGen.example = example


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot(self, model=None, plot_col='FlowHt', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        #  for n in range(3, 32):
        plt.subplot(3, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('IR')

WindowGenerator.plot = plot

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_batch(self, model=None, dset_name=None, plot_col='FlowHt', max_subplots=40, n_batch=None,
               figures=None, xy_fig=None):

  font = { 'family': 'serif', 'color' : 'black', 'weight': 'normal', 'size'  :  10}
# for WinGen() window object(instance ?)

  if dset_name == 'train':
        dataset = self.train
  elif dset_name == 'val':
        dataset = self.val
  elif dset_name == 'test' :
        dataset = self.test
  else :
        dataset = self.train

  if model is not None :
        fig, ax2 = plt.subplots(figsize =(7,3))
        if xy_fig == 'True':
            fig, ax3 = plt.subplots(figsize =(5,5))
            #ax3.margins(0.5)

  #           ax3.plot([-10, 10], [-10, 10], 'k--') # dashed diagonal

  for i, batch in enumerate(dataset) :
#            print (i)

            inputs, labels, lab = batch
            nrows = len(inputs)
#            print (nrows)
            if nrows == 1 :nrows=2

            x_max = np.array([tf.reduce_max(labels)])
            x_min = np.array([tf.reduce_min(labels)])

            fvsize = nrows * 1.5
            plot_col_index = self.column_indices[plot_col]
            max_n = min(max_subplots, len(inputs))
            if figures == 'True':
                fig, ax = plt.subplots(nrows, ncols=1, figsize = (12,fvsize), tight_layout = True)
            for n in range(max_n):
        #           plt.plot(self.input_indices, inputs[n, :, plot_col_index],

                if figures == 'True':
                    ax[n].plot(inputs[n, :, 0], inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)
                    ax[n].text(0.9,0.1, "n = {}".format(n), fontdict =font, ha="center", transform=ax[n].transAxes)

                if self.label_columns:
                        label_col_index = self.label_columns_indices.get(plot_col, None)
                else:
                        label_col_index = plot_col_index

                if label_col_index is None:
                        continue
                if figures == 'True':
                        ax[n].scatter(lab[n,:, 0], labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

                if model is not None:
                        ax2.scatter(lab[n,:,0], labels[n, :, label_col_index],
                        marker='.',facecolors='none', edgecolors='b' )

                if model is not None:
                      predictions = model(inputs)
                      if figures == 'True':
                        ax[n].scatter(lab[n,:, 0],  predictions[n, :, label_col_index],
                                  marker='X', edgecolors='k', label='Predictions',
                                        facecolors='none',  c='#ff7f0e', s=64)
                      ax2.scatter(lab[n,:, 0],   predictions[n, :, label_col_index],
                          marker=',', label='Predictions', facecolors='none', edgecolors='r',
                                   s=16)
                      if xy_fig == 'True':
                        ax3.scatter(predictions[n, :, label_col_index], labels[n, :, label_col_index],
                                  marker='.', facecolors='none', edgecolors='b')

                        l_max = np.array([tf.reduce_max(labels)])
                        p_max = np.array([tf.reduce_max(predictions)])
                        x_max_tmp = max(l_max, p_max)
                        l_min = np.array([tf.reduce_min(labels)])
                        p_min = np.array([tf.reduce_min(predictions)])
                        x_min_tmp = max(l_min, p_min)
                        if x_max < x_max_tmp :
                            x_max = x_max_tmp
                        if x_min > x_min_tmp :
                            x_min = x_min_tmp

                      if n == 0:
                        if figures == 'True':
                           # ax[n].legend()
                            ax[n].set_ylabel(f'{plot_col} [normed]')
                     #   ax[n].set_ylim([-1.5, 0.1])
                     #   ax[n].set_xlim([-2.2, 1.12])
                      if figures == 'True':
                        ax[n].invert_xaxis()
                        ax[n].set_xlabel('IR')
            if n_batch is not None :
                    if n_batch <= i :
                        break
                    else:
                        continue
            else :
                continue

  if model is not None :
#  ax2.legend( )
#    ax3.set_ylim (self.train_df['FlowHt'].min(0)-0.2,
#                self.test_df['FlowHt'].max(0)+1.0)
    if xy_fig == 'True':
#        ax3.set_xlim(self.test_df['IR'].min(0)-0.2, self.train_df['IR'].max(0)+0.2)
#        ax3.set_ylim(self.test_df['IR'].min(0)-0.2, self.train_df['IR'].max(0)+0.2)
#        ax3.plot([self.test_df['IR'].min(0)-0.2, self.train_df['IR'].max(0)+0.2],
#                 [self.test_df['IR'].min(0)-0.2, self.train_df['IR'].max(0)+0.2], 'k--') # dashed diagonal
        y_min = tf.identity(x_min)
        y_max = tf.identity(x_max)
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.plot([x_min, y_max], [x_min, y_max], 'k--') # dashed diagonal
        # ax3.invert_xaxis()

  ax2.invert_xaxis()


WinGen.plot_batch = plot_batch


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_xy(self, model=None, dset_name=None, plot_col='FlowHt', max_subplots=40, n_batch=None,
               figures=None):
  if dset_name is not None :
      if dset_name == 'train':
            dataset = self.train
      elif dset_name == 'val':
            dataset = self.val
      elif dset_name == 'test' :
            dataset = self.test
#      else :
#            dataset = self.train

  if model is not None :
        fig, ax2 = plt.subplots(figsize =(7,3))
        ax2.margins(0.01)
#  fig, ax4 = plt.subplots(figsize=(7, 3))

  fig, ax3 = plt.subplots(figsize =(5,5))
  ax3.margins(0.01)

  markers = ["o", "s", "D", "h"]
  colors = ["blue", "red", "green", "k",  "m", 'cyan', 'y', 'w']

  if dset_name is not None :
      data = [dataset]
  else :
      data = [self.train, self.val]
      #data = [self.train, self.val, self.test]
  for j, dataset in enumerate(data) :
      for i, batch in enumerate(dataset) :
            inputs, labels, lab = batch
#            nrows = len(inputs)
#            if nrows == 1 :nrows=2
            if  j == 0 and i == 0 :
              x_max_0 = np.array([tf.reduce_max(labels)])
              x_min_0 = np.array([tf.reduce_min(labels)])

            x_max_tmp = np.array([tf.reduce_max(labels)])
            x_min_tmp = np.array([tf.reduce_min(labels)])
            if x_max_0 <= x_max_tmp:
                x_max = x_max_tmp
                x_max_0 = x_max_tmp
            if x_min_0 >= x_min_tmp:
                x_min = x_min_tmp
                x_min_0 = x_min_tmp

            if model is not None:
                predictions = model(inputs)

            #            fvsize = nrows * 1.5
            plot_col_index = self.column_indices[plot_col]
            max_n = min(max_subplots, len(inputs))
#            if figures == 'True':
#                fig, ax = plt.subplots(nrows, ncols=1, figsize = (12,fvsize), tight_layout = True)
#            ax2.plot(inputs[:, :, 0], inputs[:, :, plot_col_index],label='Inputs', marker='.', zorder=-10)
#            ax4.plot(inputs[:, :, 0], inputs[:, :, plot_col_index],label='Inputs', marker='.', zorder=-10)
            ax2.scatter(inputs[:, :, 0], inputs[:, :, plot_col_index],label='Inputs', marker='.',
                        color=colors[j], zorder=-10, facecolors='none')
      #      ax4.scatter(inputs[:, :, 0], inputs[:, :, plot_col_index],label='Inputs', marker='.' )

            for n in range(max_n):
#                if figures == 'True':
#                    ax[n].plot(inputs[n, :, 0], inputs[n, :, plot_col_index],
#                    label='Inputs', marker='.', zorder=-10)
#                    ax[n].text(0.9,0.1, "n = {}".format(n), fontdict =font, ha="center", transform=ax[n].transAxes)
                if self.label_columns:
                        label_col_index = self.label_columns_indices.get(plot_col, None)
                else:
                        label_col_index = plot_col_index
                if label_col_index is None:
                        continue
#                if figures == 'True':
#                        ax[n].scatter(lab[n,:, 0], labels[n, :, label_col_index],
#                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
                if model is not None:
                      #ax2.scatter(lab[n,:,0], labels[n, :, label_col_index],
                      #       marker = 'o', color = colors[j+4], facecolors='none',   zorder=2,
                      #          label='Labels' )

                      #ax2.plot(lab[n,:,0], labels[n, :, label_col_index], 'r-', zorder=1.5 )

                     # ax2.scatter(lab[n,:, 0],   predictions[n, :, label_col_index], label='Predictions',
                      ax2.scatter(lab[n, :, 0], predictions[n, :, plot_col_index], label='Predictions',
                                              marker='o',  color = 'red', facecolors='none', s=8)

                      ax3.scatter(predictions[n, :, plot_col_index], labels[n, :, label_col_index],
                                color = colors[j],  facecolors='none', marker = '.')

                      l_max = np.array([tf.reduce_max(labels)])
                      p_max = np.array([tf.reduce_max(predictions)])
                      x_max_tmp = max(l_max, p_max)
                      l_min = np.array([tf.reduce_min(labels)])
                      p_min = np.array([tf.reduce_min(predictions)])
                      x_min_tmp = max(l_min, p_min)
                      if x_max_0 < x_max_tmp :
                        x_max = x_max_tmp
                        x_max_0 = x_max_tmp
                      if x_min_0 > x_min_tmp :
                        x_min = x_min_tmp
                        x_min_0 = x_min_tmp

 #           ax4.scatter(lab[:, :, label_col_index], labels[:, :, label_col_index],
 #                 edgecolors='g', c='#2ca02c', s=25, zorder=2, label='Labels')
            #ax4.plot(lab[:, :, label_col_index], labels[:, :, label_col_index], 'r-', zorder=1.5)
#            ax4.scatter(lab[:, :, label_col_index], predictions[:, :, label_col_index], label='Predictions',
#                  marker=',', edgecolors='k', facecolors='none', c='#ff7f0e', s=16, zorder=3)

            if n_batch is not None :
                    if n_batch <= i :
                        break
                    else:
                        continue
            else :
                continue
            x_max_0 = x_max.copy()
            x_min_0 = x_min.copy()
 #     if j==0 and i==0 :
 #       ax4.legend()

#    ax3.set_ylim (self.train_df['FlowHt'].min(0)-0.2,
#                self.test_df['FlowHt'].max(0)+1.0)
#    if xy_fig == 'True':
#        ax3.set_xlim(self.test_df['IR'].min(0)-0.2, self.train_df['IR'].max(0)+0.2)
#        ax3.set_ylim(self.test_df['IR'].min(0)-0.2, self.train_df['IR'].max(0)+0.2)
#        ax3.plot([self.test_df['IR'].min(0)-0.2, self.train_df['IR'].max(0)+0.2],
#                 [self.test_df['IR'].min(0)-0.2, self.train_df['IR'].max(0)+0.2], 'k--') # dashed diagonal
  if model is not None:
        y_min = tf.identity(x_min)
        y_max = tf.identity(x_max)
#        ax3.set_xlim(x_min, x_max)
#        ax3.set_ylim(y_min, y_max)
        ax3.plot([x_min, y_max], [x_min, y_max], 'k--') # dashed diagonal
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Measured')


# ax3.invert_xaxis()

  ax2.invert_xaxis()
#  ax4.invert_xaxis()

  return(ax2, ax3)
#  if i == 0: ax2.legend()

WinGen.plot_xy = plot_xy



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_4(self, model=None, plot_col='FlowHt', max_subplots=40, n_batch=None) :
#def plot_batch(self, model=None, dset_name=None, plot_col='FlowHt', max_subplots=40, n_batch=None,
#                   figures=None, xy_fig=None):
    ax = plt.gca()

    for i, batch in enumerate(self.train) :

            inputs, labels, lab = batch
            nrows = len(inputs)
            predictions = model(inputs)

            plot_col_index = self.column_indices[plot_col]
            max_n = min(max_subplots, len(inputs))
            for n in range(max_n):
                ax.plot(inputs[n, :, 0], inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)
            #    ax.text(0.9,0.1, "n = {}".format(n), fontdict =font, ha="center", transform=ax[n].transAxes)
                if self.label_columns:
                        label_col_index = self.label_columns_indices.get(plot_col, None)
                else:
                        label_col_index = plot_col_index
                if label_col_index is None:
                        continue

                ax.scatter(lab[n,:, 0], labels[n, :, label_col_index],
                           edgecolors='k', label='Labels', c='#2ca02c', s=64)

                # predictions = model(inputs)
                ax.scatter(lab[n,:, 0],  predictions[n, :, label_col_index],
                                  marker='X', edgecolors='k', label='Predictions',
                                        facecolors='none',  c='#ff7f0e', s=64)

                self.train_df.iloc[i*n][plot_col_index] = predictions[n, :, label_col_index]


    ax.invert_xaxis()


WindowGenerator.plot_4 = plot_4


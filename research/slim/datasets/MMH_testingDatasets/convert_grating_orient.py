# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import numpy as np
import shutil

import tensorflow as tf

from slim.datasets import dataset_utils

#%% set up some useful constants and parameters

# The percntage of images in the validation set.
_PCT_VAL = 0.10

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 10

# information about the objects. There are 6 total categories, and 36 exemplars 
# of each. Each exemplar is rendered at 144 total viewpoints, with 12 steps in 
# X rotation, and 12 steps in Y rotation. 

nX=360
nY=1
nEx=2
nCat=1

# names of the categories     
#cats = [['a'],['b'],['c'],['d'],['e'],['f']]
    
# list all the image features in a big matrix, where every row is unique.
#catlist=np.expand_dims(np.transpose(np.repeat(np.arange(nCat)+1,nX*nY*nEx)),1)
exlist=np.transpose(np.tile(np.repeat(np.arange(nEx)+1,nX*nY),[1,nCat]))
xlist=np.transpose(np.tile(np.repeat(np.arange(nX),nY),[1,nEx*nCat]))
ylist=np.transpose(np.tile(np.arange(nY),[1,nX*nEx*nCat]))

featureMat = np.concatenate((exlist,xlist,ylist),axis=1)

#%%

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
#    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
#    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

#  def decode_jpeg(self, sess, image_data):
#    image = sess.run(self._decode_jpeg,
#                     feed_dict={self._decode_jpeg_data: image_data})
#    assert len(image.shape) == 3
#    assert image.shape[2] == 3
#    return image

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(image_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """  
  
    # we'll train the model on X rotation right now
  xlist_actual=np.mod(xlist,180)
  nBins = int(36)
  assert np.mod(180,nBins)==0
  nPerBin = int(180/nBins)
  binned_orient = np.reshape(np.arange(0,180,1),[nBins,nPerBin]) 

  class_names = []
  model_labels = np.zeros(np.shape(xlist_actual))
  for bb in range(nBins):
      inds = np.where(np.isin(xlist_actual, binned_orient[bb,:]))[0]
      model_labels[inds] = bb
      class_names.append('%.f_through_%.f_deg' % (binned_orient[bb,0], binned_orient[bb,nPerBin-1]))
    
     
  all_labels = []
  all_filenames = []
  for ii in np.arange(0,np.size(model_labels)):
    index = ii
#    print(index)
    full_fn = image_dir + "phase" +str(exlist[index][0]) + "/" + \
    "phase" + str(exlist[index][0]) + "_x" + str(xlist[index][0]) + "_y" + str(ylist[index][0]) + ".png"

    all_labels.append(model_labels[ii])
    all_filenames.append(full_fn)
  
  return all_filenames, all_labels, class_names, xlist

def _get_dataset_filename(dataset_dir, split_name, num_shards, shard_id):
  output_filename = 'grating_orient_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_labels, dataset_dir, num_shards):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  
  print(split_name)
  
  assert (split_name in ['train', 'validation'] or 'batch' in split_name) 

  num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(num_shards):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, num_shards, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

#            class_name = os.path.basename(os.path.dirname(filenames[i]))
            
            class_id = class_labels[i]
#            class_name = class_names[int(class_id)]
            
            print('\n %s, label %d\n' % (filenames[i],class_id))
            
            example = dataset_utils.image_to_tfexample(
                image_data, b'png', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


#def _clean_up_temporary_files(dataset_dir):
#  """Removes temporary files used to create the dataset.
#
#  Args:
#    dataset_dir: The directory where the temporary files are stored.
#  """
#  filename = _DATA_URL.split('/')[-1]
#  filepath = os.path.join(dataset_dir, filename)
#  tf.gfile.Remove(filepath)
#
#  tmp_dir = os.path.join(dataset_dir, 'flower_photos')
#  tf.gfile.DeleteRecursively(tmp_dir)


#def _dataset_exists(dataset_dir):
#  for split_name in ['train', 'validation','allims']:
#    for shard_id in range(_NUM_SHARDS):
#      output_filename = _get_dataset_filename(
#          dataset_dir, split_name, shard_id)
#      if not tf.gfile.Exists(output_filename):
#        return False
#  return True


def main(argv):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  
  dataset_dir = '/usr/local/serenceslab/maggie/biasCNN/datasets/datasets_Grating_Orient/'
  image_dir = '/usr/local/serenceslab/maggie/biasCNN/grating_ims/'
  
  # if the folder already exists, we'll automatically delete it and make it again.
  if tf.gfile.Exists(dataset_dir):
    print('deleting')
#        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    shutil.rmtree(dataset_dir, ignore_errors = True)
    tf.gfile.MkDir(dataset_dir)
  else:
    tf.gfile.MkDir(dataset_dir)

 
#%% get the information for ALL my images (all categories, exemplars, rotations)
    
  all_filenames, all_labels, class_names, orig_angles = _get_filenames_and_classes(image_dir)
 
# save out this list just as a double check that this original order is correct
  np.save(dataset_dir + 'all_filenames.npy', all_filenames)
  np.save(dataset_dir + 'all_labels.npy', all_labels)
  np.save(dataset_dir + 'orig_angles.npy', orig_angles)
#%% Define my training and validation sets. 
# Random 10 percent is validation

  random.seed(_RANDOM_SEED)   
  
  fullseq = np.arange(0,np.size(all_labels))
  random.shuffle(fullseq)
  
  num_val = int(np.ceil(np.size(all_labels)*_PCT_VAL))
  
  valinds_num = fullseq[:num_val]
  trninds_num = fullseq[num_val:]
  
  training_filenames = []
  validation_filenames = []
  training_labels = []
  validation_labels=[]
    
  for ii in trninds_num:
      training_filenames.append(all_filenames[ii])
      training_labels.append(all_labels[ii])
    
  for ii in valinds_num:
      validation_filenames.append(all_filenames[ii])
      validation_labels.append(all_labels[ii])
           
 

    
  # First, convert the training and validation sets. these will be automatically
  # divided into num_shards (5 sets), which speeds up the training procedure.
  _convert_dataset('train', training_filenames, training_labels, dataset_dir,num_shards=_NUM_SHARDS)
  _convert_dataset('validation', validation_filenames, validation_labels, dataset_dir,num_shards=_NUM_SHARDS)

  # Second - save the validation set as a couple "batches" of images. 
  # Doing this manually makes it easy to load them and get their weights later on. 
  n_total_val = np.size(all_labels)
  max_per_batch = int(360);
  num_batches = np.ceil(n_total_val/max_per_batch)
  
  for bb in np.arange(0,num_batches):
     
      bb=int(bb)
      name = 'batch' + str(bb)
      
      if (bb+1)*max_per_batch > np.size(all_filenames):
          batch_filenames = all_filenames[bb*max_per_batch:-1]
          batch_labels = all_labels[bb*max_per_batch:-1]
      else:     
          batch_filenames =all_filenames[bb*max_per_batch:(bb+1)*max_per_batch]
          batch_labels = all_labels[bb*max_per_batch:(bb+1)*max_per_batch]

          assert np.size(batch_labels)==max_per_batch

      _convert_dataset(name, batch_filenames, batch_labels, dataset_dir,num_shards=1)
  
      np.save(dataset_dir + name + '_filenames.npy', batch_filenames)
      np.save(dataset_dir + name + '_labels.npy', batch_labels)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

#  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the grating dataset, with orientation labels!')


if __name__ == "__main__":
  tf.app.run()
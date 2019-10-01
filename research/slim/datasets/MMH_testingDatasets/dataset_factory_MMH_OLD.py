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
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import cifar10
from datasets import flowers
from datasets import imagenet
from datasets import mnist
from datasets import objects_view
from datasets import objects_category
from datasets import objects_view_category1
from datasets import grating_orient
from datasets import grating_orient_smallbatches
from datasets import grating_orient_sf
from datasets import grating_orient_sf_withnoise

datasets_map = {
    'cifar10': cifar10,
    'flowers': flowers,
    'imagenet': imagenet,
    'mnist': mnist,
    'objects_view': objects_view,
    'objects_category':objects_category,
    'objects_view_category1':objects_view_category1,
    'grating_orient':grating_orient,
    'grating_orient_smallbatches':grating_orient_smallbatches,
    'grating_orient_sf':grating_orient_sf,
    'grating_orient_sf_withnoise':grating_orient_sf_withnoise,

}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  return datasets_map[name].get_split(
      split_name,
      dataset_dir,
      file_pattern,
      reader)

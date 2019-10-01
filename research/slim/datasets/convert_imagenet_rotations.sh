#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
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

# Script to download and preprocess ImageNet Challenge 2012
# training and validation data set.
#
# The final output of this script are sharded TFRecord files containing
# serialized Example protocol buffers. See build_imagenet_data.py for
# details of how the Example protocol buffers contain the ImageNet data.
#
# The final output of this script appears as such:
#
#   data_dir/train-00000-of-01024
#   data_dir/train-00001-of-01024
#    ...
#   data_dir/train-00127-of-01024
#
# and
#
#   data_dir/validation-00000-of-00128
#   data_dir/validation-00001-of-00128
#   ...
#   data_dir/validation-00127-of-00128
#
# Note that this script may take several hours to run to completion. The
# conversion of the ImageNet data to TFRecords alone takes 2-3 hours depending
# on the speed of your machine. Please be patient.
#
# **IMPORTANT**
# To download the raw images, the user must create an account with image-net.org
# and generate a username and access_key. The latter two are required for
# downloading the raw images.
#
# usage:
#  cd research/slim
#  bazel build :download_and_convert_imagenet
#  ./bazel-bin/download_and_convert_imagenet.sh [data-dir]
set -e

# Specify the directory i'm working in
DATA_DIR=/usr/local/serenceslab/maggie/tensorflow/models/research/slim/datasets/ILSVRC2012/
SCRIPT_DIR=/usr/local/serenceslab/maggie/tensorflow/models/research/slim/datasets/

LABELS_FILE="${SCRIPT_DIR}imagenet_lsvrc_2015_synsets.txt"
BOUNDING_BOX_FILE="${DATA_DIR}bounding_boxes/imagenet_2012_bounding_boxes.csv"
IMAGENET_METADATA_FILE="${SCRIPT_DIR}imagenet_metadata.txt"
 
declare -a rot_list=(0 22 45)

for rot in ${rot_list[@]}
do   

    # Where the training/testing sets of images are now
    TRAIN_DIRECTORY="${DATA_DIR}train_rot_${rot}/"
    VALIDATION_DIRECTORY="${DATA_DIR}validation_rot_${rot}/"
    
    # where i'll put all the tfrecord files
    OUTPUT_DIRECTORY="${DATA_DIR}tfrecord_rot_${rot}/"
    
    mkdir -p ${OUTPUT_DIRECTORY}
    
     
    # Build the TFRecords version of the ImageNet data.
    BUILD_SCRIPT="${SCRIPT_DIR}build_imagenet_data.py"
    
    python ${BUILD_SCRIPT} \
      --train_directory="${TRAIN_DIRECTORY}" \
      --validation_directory="${VALIDATION_DIRECTORY}" \
      --output_directory="${OUTPUT_DIRECTORY}" \
      --imagenet_metadata_file="${IMAGENET_METADATA_FILE}" \
      --labels_file="${LABELS_FILE}" \
      --bounding_box_file="${BOUNDING_BOX_FILE}"

done
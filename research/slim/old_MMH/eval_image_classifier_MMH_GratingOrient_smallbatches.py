# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory_MMH
from nets import nets_factory
from preprocessing import preprocessing_factory_MMH
import os
import numpy as np

slim = tf.contrib.slim

#%% define my custom paths

root = '/usr/local/serenceslab/maggie/tensorflow/novel_objects/'

dataset_dir = root + 'datasets/datasets_Grating_Orient_smallbatches'

#restore_ckpt_dir = root + 'checkpoints/inception_ckpt/inception_v3.ckpt'

#load_log_dir = root + 'logs/inception_v3_retrained_category_long'
#load_log_dir = root + 'checkpoints/inception_ckpt/inception_v3.ckpt'
#load_log_dir = root + 'logs/inception_v3_retrained_grating_orient_NOFLIP_long'
load_log_dir = root + 'logs/nasnet_retrained_grating_orient_long'
#save_eval_dir = root + 'logs/inception_v3_retrained_category_long_EVAL'
save_eval_dir = root + 'logs/nasnet_grating_orient_EVAL'

#save_weights_dir=  root + 'weights/inception_v3_grating_orient_NOFLIP_long'
save_weights_dir=  root + 'weights/nasnet_grating_orient_long'
#save_weights_dir=  root + 'weights/inception_v3_retrained_category_short'

if not os.path.isdir(save_weights_dir):
    os.mkdir(save_weights_dir)

dataset_name = 'grating_orient_smallbatches'

#split_name = 'validation'

#which_model = 'inception_v3'
which_model = 'nasnet_large'

flipLR = False

#%%

tf.app.flags.DEFINE_integer(
    'batch_size',90, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', 1,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', load_log_dir,
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', save_eval_dir, 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', dataset_name, 'The name of the dataset to load.')

#tf.app.flags.DEFINE_string(
#    'dataset_split_name', split_name, 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', dataset_dir, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', which_model, 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
      raise ValueError('You must supply the dataset directory with --dataset_dir')


  num_batches=8

  for bb in np.arange(0,num_batches):
    
      batch_name = 'batch'+str(bb)
      
#      tf.app.flags.DEFINE_string(
#              'dataset_split_name',batch_name, 'The name of the train/test split.')

      tf.logging.set_verbosity(tf.logging.INFO)
  
      with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
    
        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory_MMH.get_dataset(
            FLAGS.dataset_name, batch_name, FLAGS.dataset_dir)
    
        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)
    
        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=1,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset
    
    
        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory_MMH.get_preprocessing(
            preprocessing_name,
            is_training=False, flipLR=False)
    
        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
    
        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
    
        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)
    
    #    ims_orig = tf.identity(images);
    #    labels_orig = tf.identity(labels);
    
        ####################
        # Define the model #
        ####################
        logits, end_pts = network_fn(images)
        
            
        
        if FLAGS.moving_average_decay:
          variable_averages = tf.train.ExponentialMovingAverage(
              FLAGS.moving_average_decay, tf_global_step)
          variables_to_restore = variable_averages.variables_to_restore(
              slim.get_model_variables())
          variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
          variables_to_restore = slim.get_variables_to_restore()
    
        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)
    
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5),
        })
    
        # Print the summaries to screen.
        for name, value in names_to_values.items():
          summary_name = 'eval/%s' % name
          op = tf.summary.scalar(summary_name, value, collections=[])
          op = tf.Print(op, [value], summary_name)
          tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    
        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
          num_batches = FLAGS.max_num_batches
        else:
          # This ensures that we make a single pass over all of the data.
          num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
    
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
          checkpoint_path = FLAGS.checkpoint_path
    
        tf.logging.info('Evaluating %s' % checkpoint_path)
    
        out = slim.evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=checkpoint_path,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            final_op={'logits':logits, 'end_pts':end_pts,'images':images,'labels':labels,'predictions':predictions},
            variables_to_restore=variables_to_restore)
    
        end_pts= out['end_pts']
     
        logits = out['logits']
        
        prelogits = end_pts['global_pool']
        
        images = out['images']
        
        labels = out['labels']
#        print(np.max(labels))
        predictions = out['predictions']
        # this is the very last layer before logit conversion
#        lastlayer_name = 'PreLogits'
        
#        lastlayer_weights =end_pts[lastlayer_name]
       
        fn2save = save_weights_dir + '/' + batch_name + '_logits.npy'
        np.save(fn2save, logits)
          
        fn2save = save_weights_dir + '/' + batch_name + '_prelogits.npy'
        np.save(fn2save, prelogits)
           
        fn2save = save_weights_dir + '/' + batch_name + '_ims_orig.npy'
        np.save(fn2save, images)
        
        fn2save = save_weights_dir + '/' + batch_name + '_labels_orig.npy'
        np.save(fn2save, labels)
          
        fn2save = save_weights_dir + '/' + batch_name + '_labels_predicted.npy'
        np.save(fn2save, predictions)
         
                                    
if __name__ == '__main__':
  tf.app.run()

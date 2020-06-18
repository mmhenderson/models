#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:54:00 2019

@author: mmhender
"""
#import tensorflow as tf
from tensorflow.contrib.slim.python.slim import learning
#from tensorflow.contrib.slim.python.slim import evaluation

def train_step_fn(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.
      (Wrapper function for train_step in learning.py)
  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments (using this here to 
    pass in my eval_op for validation set evaluation, so we can do evaluation
    during training).

  Returns:
    The total loss and a boolean indicating whether or not to stop training.

  """
  
  total_loss, should_stop = learning.train_step(sess, train_op, global_step, train_step_kwargs)
  
  should_val = sess.run(train_step_kwargs['should_val'])
  global_step = sess.run(global_step)
  
  if should_val:
      # validate

      print('validating model on step %d'%global_step)
      sess.run([train_step_kwargs['eval_op']] )

#  train_step_kwargs['curr_step'] += 1
  
  return [total_loss, should_stop]
  
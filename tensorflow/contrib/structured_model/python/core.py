from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import collections

from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_util, sparse_ops
from tensorflow.python.platform import tf_logging as logging

_structured_model = None
_structured_model_v2 = None
__SEQUENCE_LABEL__ = "__seq_label__"

def get_structured_model():
  if _structured_model_v2:
    return _structured_model_v2
  return _structured_model

def enable_structured_model_mode(fg, mc, features, label,
                                 user_tensor, item_tensor, query_tensor=None,
                                 scope=None):
  global _structured_model
  _structured_model = StructuredModel(fg, mc, features, label,
                                      user_tensor, item_tensor, query_tensor,
                                      scope=scope)
  logging.info("enable structured_model mode, %s", _structured_model)

def enable_structured_model_mode_v2(user_tensors, item_tensors, item_seq_length):
  global _structured_model_v2
  _structured_model_v2 = StructuredModelV2(user_tensors, item_tensors, item_seq_length)
  logging.info("enable structured_model mode, %s", _structured_model_v2)

def find_boundery_tensors(user_ops, item_ops):
  queue_item = collections.deque()
  queue_item.extend(item_ops)
  queue_item_back = collections.deque()
  item_sets = set()
  processed_item = set()
  logging.info("queue_item: %s", queue_item)
  while queue_item:
    op = queue_item.popleft()
    if control_flow_util.IsInWhileLoop(op):
      input_ops = []
      for t in op.inputs:
        input_ops.append(t.op)
      queue_item_back.extend(input_ops)
    if op in processed_item:
      continue
    processed_item.add(op)
    logging.info("op.outputs: %s", op.outputs)
    for t in op.outputs:
      logging.info("t.consumers(): %s", t.consumers())
      item_sets = (item_sets|set(t.consumers()))
      queue_item.extend(t.consumers())
  logging.info("item_sets size: %d", len(item_sets))
  logging.info("adding all CFG relative ops")
  processed_item = set()
  while queue_item_back:
    op = queue_item_back.popleft()
    if op in processed_item:
      continue
    if control_flow_util.IsInWhileLoop(op):
      input_ops = []
      for t in op.inputs:
        input_ops.append(t.op)
      for t in op.outputs:
        input_ops.append(t.op)
      queue_item_back.extend(input_ops)
      item_sets.add(op)
      processed_item.add(op)
  logging.info("updated item_sets size: %d", len(item_sets))

  queue_user = collections.deque()
  queue_user.extend(user_ops)
  user_sets = set()
  boundery_tensor_sets = set()
  while queue_user:
    op = queue_user.popleft()
    for t in op.outputs:
      for op2 in t.consumers():
        if op2 in user_sets:
          continue
        if op2 in item_sets:
          logging.info("add to boundery_tensor_sets: %s, directed to %s", t, op2)
          boundery_tensor_sets.add(t)
        else:
          user_sets.add(op2)
          queue_user.append(op2)
  logging.info("user_sets size: %d", len(user_sets))
  logging.info("boundery_tensor_sets size: %d", len(boundery_tensor_sets))
  logging.info("boundery_tensor_sets: %s", boundery_tensor_sets)
  return user_sets, item_sets, boundery_tensor_sets

def add_tile_op(boundery_tensor_sets, fg, seq_mask_reshaped, user_sets):
  for t in boundery_tensor_sets:
    if not t.consumers():
      continue
    user_expand = array_ops.expand_dims(t, 1)
    user_tiled = array_ops.tile(user_expand, [1, fg.item_seq_length, 1])
    user_2d = array_ops.reshape(user_tiled, [-1, user_tiled.get_shape()[2]])
    seq_user_input = array_ops.boolean_mask(user_2d, seq_mask_reshaped)
    logging.info("add_tile_op: %s, %s", t, t.consumers())
    t_ops = copy.copy(t.consumers())
    for op in t_ops:
      if op is user_expand.op or op in user_sets:
        continue
      for index, input_t in enumerate(op.inputs):
        if input_t is t:
          op._update_input(index, seq_user_input)
          logging.info("add_tile_op detail: %s, %s, %s", op, index, seq_user_input)

def add_split_op(fg, mc_columns, features, item_tensor,
                 seq_mask_reshaped, scope):
  if not item_tensor.consumers():
    return
  seq_features = fg.sequence_features(features, mc_columns,
                                      fg.item_seq_length,
                                      fg.item_seq_name,
                                      False)
  column = fg.feature_columns_from_name(mc_columns)
  from tensorflow.contrib import layers
  seq_layer = layers.input_from_feature_columns(seq_features,
                                                column, default_id=0,
                                                scope=scope)

  seq = array_ops.split(seq_layer, fg.item_seq_length, axis=0)
  seq_stack = array_ops.stack(values=seq, axis=1)
  seq_2d = array_ops.reshape(seq_stack, [-1, seq_stack.get_shape()[2]])
  seq_item = array_ops.boolean_mask(seq_2d, seq_mask_reshaped)

  t = item_tensor
  logging.info("add_split_op: %s, %s", t, t.consumers())
  t_ops = copy.copy(t.consumers())
  for op in t_ops:
    for index, input_t in enumerate(op.inputs):
      if input_t is t:
        op._update_input(index, seq_item)
        logging.info("add_split_op detail: %s, %s, %s", op, index, seq_item)

def add_label_op(fg, features, label_tensor, seq_mask_reshaped):
  if not label_tensor.consumers():
    return
  raw_seq_label = array_ops.reshape(sparse_ops.sparse_tensor_to_dense(
                                      features['itm_click_seq'],
                                      default_value="0"),
                                    [-1, 1])
  default_values = [constant_op.constant([0.0], dtype=dtypes.float32)
                      for i in range(0, fg.item_seq_length)]
  seq_labels = parsing_ops.decode_csv(raw_seq_label,
                                      record_defaults=default_values,
                                      field_delim=';')
  seq_labels = array_ops.transpose(seq_labels)
  item_seq_label_2d = array_ops.reshape(seq_labels, [-1, 1])
  seq_label = array_ops.boolean_mask(item_seq_label_2d, seq_mask_reshaped)
  ops.add_to_collection(__SEQUENCE_LABEL__, seq_label)
  logging.info("add_label_op: %s, %s", label_tensor, label_tensor.consumers())
  t_ops = copy.copy(label_tensor.consumers())
  for op in t_ops:
    for index, input_t in enumerate(op.inputs):
      if input_t is label_tensor:
        op._update_input(index, seq_label)
        logging.info("add_label_op detail: %s, %s, %s", op, index, seq_label)

class StructuredModel(object):

  def __init__(self, fg, mc, features, label,
               user_tensor, item_tensor, query_tensor=None,
               scope=None):
    self.fg = fg
    self.mc = mc
    self.features = features
    self.label = label
    self.user_tensor = user_tensor
    self.item_tensor = item_tensor
    self.query_tensor = query_tensor
    self.scope = scope

  def graph_transform(self):
    """
    Graph transform
    ====> Behavior
    first time call graph_transform()
      Graph:
        user_tenosr_subGraph ----> user_net ------------\
                                                         -----> other_net
        non-user_tenosr_subGraph ----> non-user_net ----/

                                       ||
                                       ||
                                       ||
                                       \/

        user_tenosr_subGraph ----> user_net ----> subGraph1 ----\
                                                                 ----> other_net
        subGraph2 ----> non-user_net ---------------------------/
      Detail:
        a .build 'sequence_mask' subGraph
        b. find boundery_tensor which collect user-subGraph and non-user-subGraph
        c. build subGraph_1 after boundery_tensor
        d. build subGraph_2 for non-user tensor(item/label) to replace former graph
    second time call graph_transform()
        Redo d. build subGraph_2 for non-user tensor(item/label) to replace former graph
        ATTENTION: DONNOT do c. because boundery_tensor is in backprop subGraph

    ====> Build Graph
    (build graph start)
    build_input()
    build_forward()
    build_loss()
    (call StructuredModel.graph_transform for forward graph transform)
    build_backprop()    (create optimizer, and compute grad. & apply grad.)
    build_other_graph()    (add summary info.)
    (call StructuredModel.graph_transform )
    (build graph end)
    """
    logging.info("graph_transform: start")
    user_op = self.user_tensor.op
    item_op = self.item_tensor.op
    query_op = self.query_tensor.op

    seq_mask_reshaped = ops.get_collection("_seq_mask_reshaped")
    if not seq_mask_reshaped:
      itm_seq_length = self.features["itm_seq_length"]
      item_seq_mask = array_ops.sequence_mask(array_ops.reshape(itm_seq_length, [-1]),
                                              self.fg.item_seq_length)
      seq_mask_reshaped = array_ops.reshape(item_seq_mask, [-1])
      ops.add_to_collection("_seq_mask_reshaped", seq_mask_reshaped)

      user_op_sets, _, boundery_tensor_sets = find_boundery_tensors(
          user_ops=[user_op],
          item_ops=[item_op] + [query_op])
      add_tile_op(boundery_tensor_sets, self.fg, seq_mask_reshaped, user_op_sets)

    seq_mask_reshaped = ops.get_collection("_seq_mask_reshaped")[0]
    add_split_op(self.fg, self.mc.item_columns, self.features,
                 self.item_tensor, seq_mask_reshaped, scope=self.scope)
    if query_op:
      add_split_op(self.fg, self.mc.query_columns, self.features,
                   self.query_tensor, seq_mask_reshaped, scope=self.scope)

    add_label_op(self.fg, self.features, self.label, seq_mask_reshaped)
    logging.info("graph_transform: end")

def is_shape_op(op):
  return op.type=="Shape" or op.type == "ShapeN" or op.type == "Rank" or op.type == "Size"

def add_tile_op_v2(boundery_tensor_sets, item_seq_length, user_sets, seq_mask_reshaped=None):
  # For training process, we cannot known the exact size of a item sequence,
  # so we need a masking. However, we do not need masking for serving as we
  # can safely assume the maximum size of the query items are item_seq_length.
  tiled_num = 0
  for t in boundery_tensor_sets:
    if not t.consumers():
      continue
    # only add tiles for those with batched tensors (dynamic shaped [?, ...])
    # as some constant operations, such as reshape, should not be tiled
    if len(t.get_shape().as_list())>0 and not t.get_shape().as_list()[0]:
      logging.info("add_tile_op [%d]: %s, %s", tiled_num, t, t.consumers())
      user_expand = array_ops.expand_dims(t, 1)
      tile_shape = [1, item_seq_length] + [1 for i in range(len(t.get_shape()[1:]))]
      user_tiled = array_ops.tile(user_expand, tile_shape)
      reshape_shape = [-1]
      if len(user_tiled.get_shape())>2:
        user_tiled_shape = array_ops.shape(user_tiled)
        reshape_shape = array_ops.concat([math_ops.reduce_prod(user_tiled_shape[:2],keepdims=True), user_tiled_shape[2:]], 0)
      seq_user_input = array_ops.reshape(user_tiled, reshape_shape)
      if seq_mask_reshaped:
        seq_user_input = array_ops.boolean_mask(seq_user_input, seq_mask_reshaped)
      
      t_ops = copy.copy(t.consumers())
      for op in t_ops:
        if not is_shape_op(op):
          if op is user_expand.op or op in user_sets:
            continue
        for index, input_t in enumerate(op.inputs):
          if input_t is t:
            op._update_input(index, seq_user_input)
            logging.info("add_tile_op detail: %s, %s, %s", op, index, seq_user_input)
      tiled_num += 1
  logging.info("add_tile_op: total %d", tiled_num)

def add_split_op_v2(item_tensor):
  if not item_tensor.consumers():
    return
  t = item_tensor
  logging.info("add_split_op: %s, %s", t, t.consumers())
  t_ops = copy.copy(t.consumers())
  seq_item = string_ops.string_split_v2(t, '')
  seq_item = sparse_ops.sparse_tensor_to_dense(seq_item)
  # Flatten the items [user_batch * item_pack_size]
  seq_item = array_ops.reshape(seq_item, [-1])
  for op in t_ops:
    for index, input_t in enumerate(op.inputs):
      if input_t is t:
        op._update_input(index, seq_item)
        logging.info("add_split_op detail: %s, %s, %s", op, index, seq_item)

class StructuredModelV2(object):
  # user_tensors: list of common tensor
  # item_tensors: list of packed tensor
  def __init__(self, user_tensors, item_tensors, item_seq_length):
    self.user_tensors = user_tensors
    self.item_tensors = item_tensors
    self.item_seq_length = item_seq_length

  def graph_transform(self):
    """
    Graph transform
    ====> Behavior
    first time call graph_transform()
      Graph:
        user_tenosr_subGraph ----> user_net ------------\
                                                          -----> other_net
        non-user_tenosr_subGraph ----> non-user_net ----/

                                        ||
                                        ||
                                        ||
                                        \/

        user_tenosr_subGraph ----> user_net ----> subGraph1 ----\
                                                                  ----> other_net
        subGraph2 ----> non-user_net ---------------------------/
      Detail:
        a. build 'sequence_mask' subGraph
        b. find boundery_tensor which collect user-subGraph and non-user-subGraph
        c. build subGraph_1 after boundery_tensor
        d. build subGraph_2 for non-user tensor(item/label) to replace former graph
    second time call graph_transform()
        Redo d. build subGraph_2 for non-user tensor(item/label) to replace former graph
        ATTENTION: DONNOT do c. because boundery_tensor is in backprop subGraph

    ====> Build Graph
    (build graph start)
    build_input()
    build_forward()
    build_loss()
    (call structured_model_v2.graph_transform for forward graph transform)
    # TODO: support training
    # build_backprop()    (create optimizer, and compute grad. & apply grad.)
    # build_other_graph()    (add summary info.)
    # (call structured_model_v2.graph_transform )
    # (build graph end)
    """
    logging.info("graph_transform: start")
    user_ops = []
    item_ops = []
    for t in self.user_tensors:
      user_ops.append(t.op)
    for t in self.item_tensors:
      item_ops.append(t.op)

    # subGraph_1
    user_op_sets, _, boundery_tensor_sets = find_boundery_tensors(
        user_ops=user_ops,
        item_ops=item_ops)
    add_tile_op_v2(boundery_tensor_sets, self.item_seq_length, user_op_sets)

    # subGraph_2
    # for t in self.item_tensors:
    #   add_split_op_v2(t)

def create_tile_ops(user_tensor, item_seq_length):
  t = user_tensor
  logging.info("add_tile_op: %s, %s", t, t.consumers())
  user_expand = array_ops.expand_dims(t, 1)
  tile_shape = [1, item_seq_length] + [1 for i in range(len(t.get_shape()[1:]))]
  user_tiled = array_ops.tile(user_expand, tile_shape)
  reshape_shape = [-1]
  if len(user_tiled.get_shape())>2:
    user_tiled_shape = array_ops.shape(user_tiled)
    reshape_shape = array_ops.concat([math_ops.reduce_prod(user_tiled_shape[:2],keepdims=True), user_tiled_shape[2:]], 0)
  seq_user_input = array_ops.reshape(user_tiled, reshape_shape)

  return seq_user_input

def create_split_ops(item_tensor):
  if not item_tensor.consumers():
    return
  t = item_tensor
  logging.info("add_split_op: %s, %s", t, t.consumers())
  t_ops = copy.copy(t.consumers())
  seq_item = string_ops.string_split_v2(t, '')
  seq_item = sparse_ops.sparse_tensor_to_dense(seq_item)
  # Flatten the items [user_batch * item_pack_size]
  seq_item = array_ops.reshape(seq_item, [-1])
  for op in t_ops:
    for index, input_t in enumerate(op.inputs):
      if input_t is t:
        op._update_input(index, seq_item)
        logging.info("add_split_op detail: %s, %s, %s", op, index, seq_item)
  return seq_item
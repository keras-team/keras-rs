import abc
from typing import Optional, Dict, Any, Callable
from keras_rs.src import types

import keras
from keras import ops

def check_tensor_shapes(tensors):
    """Checks the tensor shapes to be compatible."""
    if not tensors:
        return
    shapes = [ops.shape(ops.convert_to_tensor(tensor)) for tensor in tensors]
    
    # Checking the tensors should have rank 2
    for i, shape in enumerate(shapes):
        if len(shape) != 2:
            raise ValueError(f"Tensor {i} must have rank 2, got rank {len(shape)}")
    
    # Checking the tensor shapes are equal
    reference_shape = shapes[0]
    for i, shape in enumerate(shapes[1:], 1):
        if not ops.all(ops.equal(shape, reference_shape)):
            raise ValueError(f"Tensor {i} shape {shape} incompatible with reference shape {reference_shape}")

def apply_pairwise_op(
  x: types.Tensor, op: Callable[[types.Tensor, types.Tensor], types.Tensor]) -> types.Tensor:
  return op(
        ops.expand_dims(x, axis=-1),
        ops.expand_dims(x, axis=-2),)

def is_label_valid(labels):
  """Returns a boolen tensor, indicating whether the labels are valid."""
  labels = ops.convert_to_tensor(labels)
  return ops.greater_equal(labels, 0.)

def get_valid_pairs_and_clean_labels(labels):
  """Returns a boolean Tensor for valid pairs and cleaned labels."""
  labels = ops.convert_to_tensor(labels)
  
  # Check that labels has rank 2
  labels_shape = ops.shape(labels)
  if len(labels_shape) != 2:
    raise ValueError(f"Expected labels to have rank 2, but got rank {len(labels_shape)}")
  
  is_valid = is_label_valid(labels)
  
  valid_pairs = apply_pairwise_op(is_valid, ops.logical_and)
  labels = ops.where(is_valid, labels, ops.zeros_like(labels))
  return valid_pairs, labels

def _get_shuffle_indices(shape, mask=None, shuffle_ties=False, seed=None):

  # Produces random values when ties are to be shuffled, otherwise all zeros.
  if shuffle_ties:
    shuffle_values = keras.random.uniform(shape, seed=seed)
  else:
    shuffle_values = ops.zeros(shape, dtype="float32")

  # Given shuffle_values are consistently within [0, 1), we can safely augment
  # entries corresponding to mask=False by 2.0. This ensures their placement
  # at the end during the argsort operation.
  if mask is not None:
    shuffle_values = ops.where(mask, shuffle_values, shuffle_values + 2.0)

  # Determines indices by performing an argsort on the shuffle values.
  return ops.argsort(shuffle_values, True)

def sort_by_scores(scores, features_list, topn=None):
    scores = ops.cast(scores, "float32")
    
    # Check that scores has rank 2
    scores_shape = ops.shape(scores)
    if len(scores_shape) != 2:
        raise ValueError(f"Expected scores to have rank 2, but got rank {len(scores_shape)}")

    batch_size = ops.shape(scores)[0]
    list_size = ops.shape(scores)[1]

    if topn is None:
      topn = list_size
    topn = ops.minimum(topn, list_size)

    # Get top-k indices
    _, indices = ops.top_k(scores, topn, sorted=True)  # [B, topn]

    # Now gather features using manual indexing
    sorted_features = []
    for feat in features_list:
        # feat: [B, list_size]
        batch_indices = ops.arange(batch_size)[:, None]  # [B, 1]
        batch_indices = ops.repeat(batch_indices, topn, axis=1)  # [B, topn]
        gather_indices = ops.stack([batch_indices, indices], axis=-1)  # [B, topn, 2]

        # Reshape to flat indexing
        feat_flat = ops.reshape(feat, [-1])
        batch_indices_flat = ops.reshape(gather_indices[:, :, 0], [-1])
        list_indices_flat = ops.reshape(gather_indices[:, :, 1], [-1])
        flat_index = batch_indices_flat * list_size + list_indices_flat

        gathered = ops.take(feat_flat, flat_index)
        gathered = ops.reshape(gathered, [batch_size, topn])
        sorted_features.append(gathered)

    return sorted_features

def inverse_max_dcg(labels,
                    gain_fn=lambda labels: ops.power(2.0, labels) - 1.,
                    rank_discount_fn=lambda rank: 1. / ops.log1p(rank),
                    topn=None):
    ideal_sorted_labels, = sort_by_scores(labels, [labels], topn=topn)
    rank = ops.arange(ops.shape(ideal_sorted_labels)[1]) + 1  # shape: (list_size,)
    rank = ops.cast(rank, dtype="float32")

    # Fix broadcasting: shape (1, list_size)
    discount = ops.expand_dims(rank_discount_fn(rank), axis=0)

    # Shape now compatible: (batch_size, list_size)
    discounted_gain = gain_fn(ideal_sorted_labels) * discount

    discounted_gain = ops.sum(discounted_gain, axis=1, keepdims=True)
    return ops.where(
        ops.greater(discounted_gain, 0.),
        1. / discounted_gain,
        ops.zeros_like(discounted_gain)
    )

def log2_inverse(ranks):
    ranks_float = ops.cast(ranks, dtype="float32")
    return 1.0 / (ops.log(ranks_float + 1.0) / ops.log(2.0))

class LambdaWeight(abc.ABC):
  
    """This interface is for ranking metric optimization using weights within 
    the LambdaLoss framework (https://ai.google/research/pubs/pub47258). 
    Implementations of this interface provide concrete lambda weight models 
    that can be used with standard losses like logistic loss and softmax loss.
    
    This implementation is compatible with TensorFlow, JAX, and PyTorch, 
    operating across these backends through the unified Keras 3 API
    """

    @abc.abstractmethod
    def pair_weights(self, labels, ranks):
        """
        Returns pairwise weights for ranking loss.

        Args:
            labels: Tensor of shape [batch_size, list_size] with relevance labels
            ranks: Tensor of shape [batch_size, list_size] with current ranks (1-based)

        Returns:
            A tensor that can weight example pairs with shape
            [batch_size, list_size, list_size].
        """
        raise NotImplementedError('Calling an abstract method.')
    
    @abc.abstractmethod
    def individual_weights(self, labels, ranks):
        """Returns the weight tensor for individual examples.

        Args:
            labels: A dense tensor of labels with shape [batch_size, list_size].
            ranks: A dense tensor of ranks with the same shape as `labels` that are
                sorted by logits.

        Returns:
            A tensor that can weight individual examples with shape [batch_size, list_size].
        """
        raise NotImplementedError('Calling an abstract method.') 
        raise NotImplementedError('Calling an abstract method.') 

class LabelDiffLambdaWeight(LambdaWeight):
  """A simple LambdaWeight to compute the pair label difference."""

  def pair_weights(self, labels, ranks):
    """Returns the absolute label difference for each pair."""
    return ops.abs(apply_pairwise_op(labels, ops.subtract))

class AbstractDCGLambdaWeight(LambdaWeight):
  """Abstract LambdaWeight for Discounted Cumulative Gain (DCG) metric."""

  def __init__(self,
               topn=None,
               gain_fn=lambda label: label,
               rank_discount_fn=lambda rank: 1. / rank,
               normalized=False):
    """Initializer.

    Ranks are 1-based, not 0-based.

    Args:
      topn: (int) The topn for the DCG metric.
      gain_fn: (function) Transforms labels.
      rank_discount_fn: (function) The rank discount function.
      normalized: (bool) If True, normalize weight by the max DCG.
    """
    self._topn = topn
    self._gain_fn = gain_fn
    self._rank_discount_fn = rank_discount_fn
    self._normalized = normalized

  @abc.abstractmethod
  def pair_rank_discount(self, ranks, topn):
    """Computes the rank-based discount for a pair.

    Args:
      ranks: A 2D `Tensor` for the 1-based ranks.
      topn: A scalar `Tensor` for the topn cutoff.

    Returns:
     A pairwise weights `Tensor` based on the `rank_discount_fn`.
    """
    raise NotImplementedError('Calling an abstract method.')

  def pair_weights(self, labels, ranks):
    """See `_LambdaWeight`."""
    check_tensor_shapes([labels, ranks])
    valid_pair, labels = get_valid_pairs_and_clean_labels(labels)
    gain = self._gain_fn(labels)
    if self._normalized:
      gain *= inverse_max_dcg(
            labels,
            gain_fn=self._gain_fn,
            rank_discount_fn=self._rank_discount_fn,
            topn=self._topn)
    pair_gain = apply_pairwise_op(gain, ops.subtract)
    pair_gain *= ops.cast(valid_pair, dtype="float32")

    list_size = ops.shape(labels)[1]
    topn = self._topn or list_size
    pair_weight = ops.abs(pair_gain) * self.pair_rank_discount(ranks, topn)
    # For LambdaLoss with relative rank difference, the scale of loss becomes
    # much smaller when applying LambdaWeight. This affects the training can
    # make the optimal learning rate become much larger. We use a heuristic to
    # scale it up to the same magnitude as standard pairwise loss.
    pair_weight *= ops.cast(ops.shape(labels)[1], dtype="float32")
    breakpoint()
    return pair_weight

  def individual_weights(self, labels, ranks):
    check_tensor_shapes([labels, ranks])
    labels = ops.convert_to_tensor(labels)
    labels = ops.where(
          is_label_valid(labels), labels, ops.zeros_like(labels))
    gain = self._gain_fn(labels)
    if self._normalized:
      gain *= inverse_max_dcg(
            labels,
            gain_fn=self._gain_fn,
            rank_discount_fn=self._rank_discount_fn,
            topn=self._topn)
    rank_discount = self._rank_discount_fn(ops.cast(ranks, dtype="float32"))
    return gain * rank_discount

class DCGLambdaWeight(AbstractDCGLambdaWeight):
  """LambdaWeight for Discounted Cumulative Gain metric."""

  def __init__(self,
               topn=None,
               gain_fn=lambda label: label,
               rank_discount_fn=lambda rank: 1. / rank,
               normalized=False,
               smooth_fraction=0.):
    """Initializer.

    Ranks are 1-based, not 0-based. Given rank i and j, there are two types of
    pair weights:
      u = |rank_discount_fn(|i-j|) - rank_discount_fn(|i-j| + 1)|
      v = |rank_discount_fn(i) - rank_discount_fn(j)|
    where u is the newly introduced one in LambdaLoss paper
    (https://ai.google/research/pubs/pub47258) and v is the original one in the
    LambdaMART paper "From RankNet to LambdaRank to LambdaMART: An Overview".
    The final pair weight contribution of ranks is
      (1-smooth_fraction) * u + smooth_fraction * v.

    Args:
      topn: (int) The topn for the DCG metric.
      gain_fn: (function) Transforms labels.
      rank_discount_fn: (function) The rank discount function.
      normalized: (bool) If True, normalize weight by the max DCG.
      smooth_fraction: (float) parameter to control the contribution from
        LambdaMART.
    """
    super().__init__(topn, gain_fn, rank_discount_fn, normalized)
    if not 0. <= smooth_fraction <= 1.:
      raise ValueError('smooth_fraction %s should be in range [0, 1].' %
                       smooth_fraction)
    self._smooth_fraction = smooth_fraction

  def pair_rank_discount(self, ranks, topn):
    """See `_LambdaWeight`."""

    def _discount_for_relative_rank_diff():
      """Rank-based discount in the LambdaLoss paper."""
      pair_valid_rank = apply_pairwise_op(ops.less_equal(ranks, topn), ops.logical_or
                                           )
      rank_diff = ops.cast(
          ops.abs(apply_pairwise_op(ranks, ops.subtract)), dtype="float32")
      pair_discount = ops.where(
          ops.logical_and(ops.greater(rank_diff, 0), pair_valid_rank),
          ops.abs(
              self._rank_discount_fn(rank_diff) -
              self._rank_discount_fn(rank_diff + 1)), ops.zeros_like(rank_diff))
      return pair_discount

    def _discount_for_absolute_rank():
      """Standard discount in the LambdaMART paper."""
      # When the rank discount is (1 / rank) for example, the discount is
      # |1 / r_i - 1 / r_j|. When i or j > topn, the discount becomes 0.
      rank_discount = ops.where(
          ops.greater(ranks, topn),
          ops.zeros_like(ops.cast(ranks, dtype="float32")),
          self._rank_discount_fn(ops.cast(ranks, dtype="float32")))
      pair_discount = ops.abs(apply_pairwise_op(rank_discount, ops.subtract))
      return pair_discount

    u = _discount_for_relative_rank_diff()
    v = _discount_for_absolute_rank()
    pair_discount = (1. - self._smooth_fraction) * u + self._smooth_fraction * v
    pair_mask = apply_pairwise_op(ops.less_equal(ranks, topn), ops.logical_or)
    return pair_discount * ops.cast(pair_mask, dtype="float32")

class ListMLELambdaWeight(LambdaWeight): 
    """
    Lambda weights for ListMLE (List Maximum Likelihood Estimation) loss.
    
    ListMLE optimizes the probability of generating the correct ranking order.
    It uses position-based discounting to emphasize top positions more.
    """
    
    def __init__(self, rank_discount_fn: Optional[Callable] = None):
        """
        Initialize ListMLE lambda weights.
        
        Args:
            rank_discount_fn: Function that takes ranks and returns discount weights.
                            Default is logarithmic discount (1/log2(rank+1)).
        """
        self.rank_discount_fn = rank_discount_fn or log2_inverse

       
    def _validate_inputs(self, labels, ranks):
      """Validate input tensors have correct shapes and types."""
      labels = ops.convert_to_tensor(labels)
      ranks = ops.convert_to_tensor(ranks)

      if labels.shape != ranks.shape:
        raise ValueError(f"Labels shape {labels.shape} must match ranks shape {ranks.shape}")

      # Ensure ranks are 1-based (minimum value should be 1)
      min_rank = ops.min(ranks)
      if min_rank < 1:
        raise ValueError(f"Ranks must be 1-based (minimum value is {min_rank})")

      return labels, ranks 

    def pair_weights(self, labels, ranks):
        """
        ListMLE doesn't use pairwise weights as it's a listwise method.
        Returns None to indicate this method is not applicable.
        """
        shape = ops.shape(labels)  
        return ops.zeros((shape[0], shape[1], shape[1]), dtype="float32")
    
    def individual_weights(self, labels, ranks):
        """
        Calculate individual weights for ListMLE loss.
        
        The weights are computed as rank discounts applied uniformly across all items.
        This emphasizes top positions more than lower positions.
        
        Args:
            labels: Tensor [batch_size, list_size] with relevance labels
            ranks: Tensor [batch_size, list_size] with current ranks (1-based)
            
        Returns:
            Tensor [batch_size, list_size] with position discount weights
        """
        labels, ranks = self._validate_inputs(labels, ranks)
        
        # Apply rank discount function
        rank_discount = self.rank_discount_fn(ops.cast(ranks, dtype="float32"))
        
        # Return uniform base weights scaled by rank discount
        base_weights = ops.ones_like(labels, dtype="float32")
        return base_weights * rank_discount

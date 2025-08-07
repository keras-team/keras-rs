import pytest
import keras
from keras import ops
from keras_rs.src import testing
from absl.testing import parameterized

# from keras.losses import deserialize
# from keras.losses import serialize

from keras_rs.src.losses.lambda_weights import (
    LabelDiffLambdaWeight,
    DCGLambdaWeight,
    ListMLELambdaWeight,
    apply_pairwise_op,
    is_label_valid,
    get_valid_pairs_and_clean_labels,
    sort_by_scores,
    inverse_max_dcg,
    check_tensor_shapes,
    log2_inverse,
)

class LambdaWeightsTest(testing.TestCase, parameterized.TestCase):
    """Test cases for utility functions."""

    def test_apply_pairwise_op(self):
        """Test pairwise operation application."""
        x = ops.convert_to_tensor([[1.0, 2.0, 3.0]])
        result = apply_pairwise_op(x, ops.subtract)
        expected = ops.convert_to_tensor([[[0., -1., -2.],
                                          [1., 0., -1.],
                                          [2., 1., 0.]]])
        self.assertAllClose(result, expected)

    def test_is_label_valid(self):
        """Test label validity checking."""
        labels = ops.convert_to_tensor([[2.0, 1.0, 0.0, -1.0]])
        result = is_label_valid(labels)
        expected = ops.convert_to_tensor([[True, True, True, False]])
        self.assertAllClose(result, expected)

    def test_get_valid_pairs_and_clean_labels(self):
        """Test valid pairs extraction and label cleaning."""
        labels = ops.convert_to_tensor([[2.0, 1.0, -1.0]])
        valid_pairs, clean_labels = get_valid_pairs_and_clean_labels(labels)
        
        expected_pairs = ops.convert_to_tensor([[[True, True, False],
                                                [True, True, False],
                                                [False, False, False]]])
        expected_labels = ops.convert_to_tensor([[2.0, 1.0, 0.0]])
        
        self.assertAllClose(valid_pairs, expected_pairs)
        self.assertAllClose(clean_labels, expected_labels)

    def test_check_tensor_shapes(self):
        """Test tensor shape compatibility checking."""
        tensor1 = ops.convert_to_tensor([[1.0, 2.0]])
        tensor2 = ops.convert_to_tensor([[3.0, 4.0]])
        
        # Should not raise for compatible shapes
        check_tensor_shapes([tensor1, tensor2])
        
        # Should raise for incompatible shapes
        tensor3 = ops.convert_to_tensor([[[1.0, 2.0]]])  # 3D tensor
        with self.assertRaises(ValueError):
            check_tensor_shapes([tensor1, tensor3])
            #add cooment

    def test_sort_by_scores(self):
        """Test sorting by scores functionality."""
        scores = ops.convert_to_tensor([[3.0, 1.0, 2.0]])
        features = [ops.convert_to_tensor([[30.0, 10.0, 20.0]])]
        
        sorted_features = sort_by_scores(scores, features, topn=2)
        
        # Should return top 2 features sorted by scores: [30.0, 20.0]
        expected = [ops.convert_to_tensor([[30.0, 20.0]])]
        self.assertAllClose(sorted_features[0], expected[0])

    def test_inverse_max_dcg(self):
        """Test inverse max DCG calculation."""
        labels = ops.convert_to_tensor([[2.0, 1.0, 0.0]])
        result = inverse_max_dcg(labels)
        
        expected_max_dcg = 5.239
        expected = ops.convert_to_tensor([[1.0 / expected_max_dcg]])
        
        self.assertAllClose(result, expected, atol=1e-4)

    def test_pair_weights_default(self):
        """Test default pair weights calculation."""
        labels = ops.convert_to_tensor([[2.0, 1.0, 0.0]])
        ranks = ops.convert_to_tensor([[1, 2, 3]])
        
        lambda_weight = LabelDiffLambdaWeight()
        result = lambda_weight.pair_weights(labels, ranks)
        
        expected = ops.convert_to_tensor([[[0., 1., 2.],
                                          [1., 0., 1.],
                                          [2., 1., 0.]]])
        self.assertAllClose(result, expected)

    def test_dcg_pair_weights_default(self):  
        """Test default DCG pair weights.""" 
        labels = ops.convert_to_tensor([[2.0, 1.0, 0.0]])
        ranks = ops.convert_to_tensor([[1, 2, 3]])
        scale = 3.0
        
        lambda_weight = DCGLambdaWeight()
        result = lambda_weight.pair_weights(labels, ranks) / scale
        
        expected = ops.convert_to_tensor([[[0., 1. / 2., 2. * 1. / 6.],
                                          [1. / 2., 0., 1. / 2.],
                                          [2. * 1. / 6., 1. / 2., 0.]]])
        self.assertAllClose(result, expected, atol=1e-5)

    def test_pair_weights_with_topn(self):
        """Test DCG pair weights with topn parameter."""
        labels = ops.convert_to_tensor([[2.0, 1.0, 0.0]])
        ranks = ops.convert_to_tensor([[1, 2, 3]])
        scale = 3.0
        
        lambda_weight = DCGLambdaWeight(topn=1)
        result = lambda_weight.pair_weights(labels, ranks) / scale
        
        expected = ops.convert_to_tensor([[[0., 1. / 2., 1. / 3.],
                                          [1. / 2., 0., 0.],
                                          [1. / 3., 0., 0.]]])
        self.assertAllClose(result, expected, atol=1e-5)

    def test_pair_weights_with_smooth_fraction(self):
        """Test DCG pair weights with smooth_fraction parameter."""
        labels = ops.convert_to_tensor([[2.0, 1.0, 0.0]])
        ranks = ops.convert_to_tensor([[1, 2, 3]])
        scale = 3.0
        
        lambda_weight = DCGLambdaWeight(smooth_fraction=1.0)
        result = lambda_weight.pair_weights(labels, ranks) / scale
        
        expected = ops.convert_to_tensor([[[0., 1. / 2., 2. * 2. / 3.],
                                          [1. / 2., 0., 1. / 6.],
                                          [2. * 2. / 3., 1. / 6., 0.]]])
        self.assertAllClose(result, expected, atol=1e-5)

    def test_pair_weights_with_topn_and_smooth_fraction(self):
        """Test DCG pair weights with both topn and smooth_fraction."""
        labels = ops.convert_to_tensor([[2.0, 1.0, 0.0]])
        ranks = ops.convert_to_tensor([[1, 2, 3]])
        scale = 3.0
        
        lambda_weight = DCGLambdaWeight(topn=1, smooth_fraction=1.0)
        result = lambda_weight.pair_weights(labels, ranks) / scale
        
        expected = ops.convert_to_tensor([[[0., 1., 2.],
                                          [1., 0., 0.],
                                          [2., 0., 0.]]])
        self.assertAllClose(result, expected, atol=1e-5)

    def test_pair_weights_with_invalid_labels(self):
        """Test DCG pair weights with invalid (negative) labels."""
        labels = ops.convert_to_tensor([[2.0, 1.0, -1.0]])
        ranks = ops.convert_to_tensor([[1, 2, 3]])
        scale = 3.0
        
        lambda_weight = DCGLambdaWeight()
        result = lambda_weight.pair_weights(labels, ranks) / scale
        
        expected = ops.convert_to_tensor([[[0., 1. / 2., 0.],
                                          [1. / 2., 0., 0.],
                                          [0., 0., 0.]]])
        self.assertAllClose(result, expected, atol=1e-5)

    def test_pair_weights_with_custom_gain_and_discount(self):
        """Test DCG pair weights with custom gain and discount functions."""
        labels = ops.convert_to_tensor([[2.0, 1.0]])
        ranks = ops.convert_to_tensor([[1, 2]])
        scale = 2.0
        
        lambda_weight = DCGLambdaWeight(
            gain_fn=lambda x: ops.power(2.0, x) - 1.0,
            rank_discount_fn=lambda r: 1.0 / ops.log1p(r)
        )
        result = lambda_weight.pair_weights(labels, ranks) / scale
        
        expected_discount_diff = 1.0 / ops.log(2.0) - 1.0 / ops.log(3.0)
        expected = ops.convert_to_tensor([[[0., 2.0 * expected_discount_diff],
                                          [2.0 * expected_discount_diff, 0.]]])
        self.assertAllClose(result, expected, atol=1e-5)

    def test_pair_weights_normalized(self):
        """Test DCG pair weights with normalization."""
        labels = ops.convert_to_tensor([[1.0, 2.0]])
        ranks = ops.convert_to_tensor([[1, 2]])
        scale = 2.0
        max_dcg = 2.5  # 2/1 + 1/2 = 2.5
        
        lambda_weight = DCGLambdaWeight(normalized=True)
        result = lambda_weight.pair_weights(labels, ranks) / scale
        
        expected = ops.convert_to_tensor([[[0., 1. / 2. / max_dcg],
                                          [1. / 2. / max_dcg, 0.]]])
        self.assertAllClose(result, expected, atol=1e-5)

    def test_individual_weights_default(self):
        """Test DCG individual weights."""
        labels = ops.convert_to_tensor([[1.0, 2.0]])
        ranks = ops.convert_to_tensor([[1, 2]])
        max_dcg = 2.5  # 2/1 + 1/2 = 2.5
        
        lambda_weight = DCGLambdaWeight(normalized=True)
        result = lambda_weight.individual_weights(labels, ranks)
        
        expected = ops.convert_to_tensor([[1.0 / max_dcg / 1.0, 2.0 / max_dcg / 2.0]])
        self.assertAllClose(result, expected, atol=1e-5)

    def test_individual_weights_with_invalid_labels(self):
        """Test DCG individual weights with invalid labels."""
        labels = ops.convert_to_tensor([[1.0, -1.0]])
        ranks = ops.convert_to_tensor([[1, 2]])
        
        lambda_weight = DCGLambdaWeight()
        result = lambda_weight.individual_weights(labels, ranks)
        
        expected = ops.convert_to_tensor([[1.0 / 1.0, 0.0 / 2.0]])
        self.assertAllClose(result, expected, atol=1e-5)

    def test_smooth_fraction_validation(self):
        """Test smooth_fraction parameter validation."""
        with self.assertRaises(ValueError):
            DCGLambdaWeight(smooth_fraction=-0.1)
        
        with self.assertRaises(ValueError):
            DCGLambdaWeight(smooth_fraction=1.1)
        
        DCGLambdaWeight(smooth_fraction=0.0)
        DCGLambdaWeight(smooth_fraction=0.5)
        DCGLambdaWeight(smooth_fraction=1.0)
    
    def test_individual_weights_shape(self):
        """Test that individual weights return correct shape"""
        labels = ops.convert_to_tensor([[0., 2., 1.], [1., 0., 2.]], dtype="float32")
        ranks = ops.convert_to_tensor([[1, 2, 3], [1, 2, 3]], dtype="int32")
        lambda_weight = ListMLELambdaWeight()
        weights = lambda_weight.individual_weights(labels, ranks)
        assert weights.shape == labels.shape
    
    def test_individual_weights_values(self):
        """Test that individual weights return correct values"""
        labels = ops.convert_to_tensor([[0., 2., 1.], [1., 0., 2.]], dtype="float32")
        ranks = ops.convert_to_tensor([[1, 2, 3], [1, 2, 3]], dtype="int32")
        lambda_weight = ListMLELambdaWeight()
        weights = lambda_weight.individual_weights(labels, ranks)
        
        expected = log2_inverse(ranks)
        self.assertAllClose(weights, expected, rtol=1e-6)

if __name__ == '__main__':
    pytest.main([__file__])
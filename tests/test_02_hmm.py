import pytest
import numpy as np

from smartsurge.models import HMMParams


class Test_HMMParams_ValidateArrayDimensions_01_NominalBehaviors:
    def test_returns_none_for_none_input(self):
        """Test that method returns None when input value is None."""
        hmm_params = HMMParams()
        info = type('obj', (), {'data': {'n_states': 3}})
        result = hmm_params.validate_array_dimensions(None, info=info)
        assert result is None
    
    def test_validates_1d_array_matching_n_states(self):
        """Test validation of 1D numpy array with length matching n_states."""
        hmm_params = HMMParams(n_states=3)
        test_array = np.array([0.1, 0.2, 0.7])
        info = type('obj', (), {'data': {'n_states': 3}})
        
        result = hmm_params.validate_array_dimensions(test_array, info=info)
        assert np.array_equal(result, test_array)
    
    def test_validates_2d_array_matching_n_states(self):
        """Test validation of 2D numpy array with shape (n_states, n_states)."""
        hmm_params = HMMParams(n_states=3)
        test_array = np.array([
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.5, 0.3, 0.2]
        ])
        info = type('obj', (), {'data': {'n_states': 3}})
        
        result = hmm_params.validate_array_dimensions(test_array, info=info)
        assert np.array_equal(result, test_array)

class Test_HMMParams_ValidateArrayDimensions_02_NegativeBehaviors:
    def test_raises_error_for_1d_array_wrong_length(self):
        """Test ValueError is raised when 1D array length doesn't match n_states."""
        hmm_params = HMMParams(n_states=3)
        test_array = np.array([0.1, 0.9])  # Length 2, but n_states is 3
        info = type('obj', (), {'data': {'n_states': 3}})
        
        with pytest.raises(ValueError, match="1D array must have length 3"):
            hmm_params.validate_array_dimensions(test_array, info=info)
    
    def test_raises_error_for_2d_array_wrong_shape(self):
        """Test ValueError is raised when 2D array shape doesn't match (n_states, n_states)."""
        hmm_params = HMMParams(n_states=3)
        test_array = np.array([
            [0.1, 0.9],
            [0.3, 0.7],
            [0.5, 0.5]
        ])  # Shape (3, 2), but should be (3, 3)
        info = type('obj', (), {'data': {'n_states': 3}})
        
        with pytest.raises(ValueError, match="2D array must have shape"):
            hmm_params.validate_array_dimensions(test_array, info=info)

class Test_HMMParams_ValidateArrayDimensions_03_BoundaryBehaviors:
    def test_minimum_n_states_value(self):
        """Test with arrays using minimum allowed n_states value (2)."""
        hmm_params = HMMParams(n_states=2)
        
        # Test 1D array
        test_1d = np.array([0.3, 0.7])
        info = type('obj', (), {'data': {'n_states': 2}})
        result_1d = hmm_params.validate_array_dimensions(test_1d, info=info)
        assert np.array_equal(result_1d, test_1d)
        
        # Test 2D array
        test_2d = np.array([
            [0.8, 0.2],
            [0.4, 0.6]
        ])
        result_2d = hmm_params.validate_array_dimensions(test_2d, info=info)
        assert np.array_equal(result_2d, test_2d)
    
    def test_maximum_n_states_value(self):
        """Test with arrays using maximum allowed n_states value (10)."""
        hmm_params = HMMParams(n_states=10)
        
        # Test 1D array
        test_1d = np.ones(10) / 10
        info = type('obj', (), {'data': {'n_states': 10}})
        result_1d = hmm_params.validate_array_dimensions(test_1d, info=info)
        assert np.array_equal(result_1d, test_1d)
        
        # Test 2D array
        test_2d = np.ones((10, 10)) / 10
        result_2d = hmm_params.validate_array_dimensions(test_2d, info=info)
        assert np.array_equal(result_2d, test_2d)

class Test_HMMParams_ValidateArrayDimensions_04_ErrorHandlingBehaviors:
    def test_fallback_to_default_n_states(self):
        """Test fallback to default n_states (3) when n_states not provided in info.data."""
        hmm_params = HMMParams()
        test_array = np.array([0.1, 0.2, 0.7])  # Array with length 3
        info = type('obj', (), {'data': {}})
        
        result = hmm_params.validate_array_dimensions(test_array, info=info)
        assert np.array_equal(result, test_array)
        
        # Test with 1D array of incorrect length
        wrong_array = np.array([0.5, 0.5])  # Length 2, but default n_states is 3
        with pytest.raises(ValueError, match="1D array must have length 3"):
            hmm_params.validate_array_dimensions(wrong_array, info=info)

class Test_HMMParams_ModelDump_01_NominalBehaviors:
    def test_numpy_arrays_converted_to_lists(self):
        """Test that numpy arrays are properly converted to Python lists."""
        hmm_params = HMMParams(
            n_states=3,
            initial_probs=np.array([0.1, 0.2, 0.7]),
            transition_matrix=np.array([
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.1, 0.8]
            ])
        )
        
        result = hmm_params.model_dump()
        
        # Check types and values
        assert isinstance(result['initial_probs'], list)
        assert isinstance(result['transition_matrix'], list)
        assert result['initial_probs'] == [0.1, 0.2, 0.7]
        assert result['transition_matrix'] == [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8]
        ]
    
    def test_non_numpy_values_unchanged(self):
        """Test that non-numpy values remain unchanged in the result dictionary."""
        hmm_params = HMMParams(
            n_states=5,  # Integer should remain an integer
            initial_probs=np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        )
        
        result = hmm_params.model_dump()
        
        assert result['n_states'] == 5
        assert isinstance(result['n_states'], int)

class Test_HMMParams_ModelDump_02_NegativeBehaviors:
    def test_empty_numpy_arrays(self):
        """Test handling of empty numpy arrays."""
        hmm_params = HMMParams(n_states=3)
        
        # Set empty arrays
        hmm_params.initial_probs = np.array([])
        
        result = hmm_params.model_dump()
        
        # Empty arrays should be converted to empty lists
        assert result['initial_probs'] == []
        assert isinstance(result['initial_probs'], list)
    
    def test_complex_nested_arrays(self):
        """Test with complex/nested numpy arrays."""
        hmm_params = HMMParams(n_states=3)
        
        # Create a complex numpy array
        complex_array = np.array([
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9])
        ])
        
        hmm_params.initial_probs = complex_array
        
        result = hmm_params.model_dump()
        
        # The nested arrays should be converted to lists
        assert result['initial_probs'] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert isinstance(result['initial_probs'], list)
        assert isinstance(result['initial_probs'][0], list)

class Test_HMMParams_ModelDump_03_BoundaryBehaviors:
    def test_arrays_with_special_values(self):
        """Test handling of arrays with special values (NaN, Inf)."""
        hmm_params = HMMParams(n_states=3)
        
        # Create arrays with special values
        special_array = np.array([np.nan, np.inf, -np.inf])
        hmm_params.initial_probs = special_array
        
        result = hmm_params.model_dump()
        
        # Check if special values are preserved in the lists
        assert np.isnan(result['initial_probs'][0])
        assert np.isinf(result['initial_probs'][1])
        assert np.isinf(result['initial_probs'][2]) and result['initial_probs'][2] < 0
    
    def test_large_numpy_arrays(self):
        """Test with very large numpy arrays."""
        hmm_params = HMMParams(n_states=3)
        
        # Create a large array (1000 elements)
        large_array = np.linspace(0, 1, 1000)
        
        # Monkey patch for testing purpose
        hmm_params.initial_probs = large_array
        
        result = hmm_params.model_dump()
        
        # Verify the large array was converted to a list
        assert isinstance(result['initial_probs'], list)
        assert len(result['initial_probs']) == 1000
        assert result['initial_probs'][0] == 0
        assert result['initial_probs'][-1] == 1
"""Test data handlers."""
import pytest
import tempfile
import pandas as pd
import numpy as np
from caracal.data import TabularDataHandler


class TestTabularDataHandler:
    """Test TabularDataHandler."""
    
    def test_tabular_handler_creation(self):
        """Test creating tabular data handler."""
        # Create test CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'feature1': range(10),
                'feature2': range(10, 20),
                'target': [0, 1] * 5
            })
            df.to_csv(f.name, index=False)
            path = f.name
        
        handler = TabularDataHandler(path, target_column='target')
        
        assert handler.data_type == 'tabular'
        assert handler.return_format == 'split_arrays'
        assert handler.target_column == 'target'
        
        # Clean up
        import os
        os.unlink(path)
    
    def test_tabular_load(self):
        """Test loading tabular data."""
        # Create test CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'x1': np.random.rand(100),
                'x2': np.random.rand(100),
                'y': np.random.randint(0, 2, 100)
            })
            df.to_csv(f.name, index=False)
            path = f.name
        
        handler = TabularDataHandler(path, target_column='y')
        data = handler.load(test_split=0.2, val_split=0.1)
        
        assert 'train_data' in data
        assert 'val_data' in data
        assert 'test_data' in data
        
        X_train, y_train = data['train_data']
        assert len(X_train) > 0
        assert len(y_train) == len(X_train)
        
        # Clean up
        import os
        os.unlink(path)

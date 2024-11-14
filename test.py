import unittest
import torch as pt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Models.synapticIntelligence_copy import SynapticIntelligence

# Mock Dataset for testing
class MockDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

# Mock Model for testing
class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.fc = nn.Linear(2, 1)  # Simple linear model

    def forward(self, x):
        return pt.sigmoid(self.fc(x))

class TestSynapticIntelligence(unittest.TestCase):
    def setUp(self):
        # Initialize a mock model and SynapticIntelligence instance
        self.model = MockModel()
        self.si = SynapticIntelligence(self.model)

        # Create some mock data
        inputs = pt.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=pt.float32)
        labels = pt.tensor([[0.0], [1.0]], dtype=pt.float32)
        self.dataset = MockDataset(inputs, labels)

        # Run an initial update to set prev_params and param_updates
        self.si.update_omega(self.dataset, nn.BCELoss())

    def test_consolidate(self):
        observe_range = 2
        
        # Update omega by calling consolidate
        self.si.consolidate(observe_range)

        # Verify that omega has been updated
        for name, param in self.model.named_parameters():
            self.assertIn(name, self.si.omega)
            # Ensure that omega is a tensor and has the same shape as the parameter
            self.assertIsInstance(self.si.omega[name], pt.Tensor)
            self.assertEqual(self.si.omega[name].shape, param.shape)

            # Optional: Check that omega values are reasonable (e.g., non-negative)
            self.assertTrue((self.si.omega[name] >= 0).all(), f"{name} omega values are negative")

    def tearDown(self):
        del self.si
        del self.model

if __name__ == '__main__':
    unittest.main()

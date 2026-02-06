import unittest
import torch
import torch.nn as nn
from src.engine.memory_manager import MemoryManager

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(32, 32)

class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.mm = MemoryManager()
        self.model = SimpleModel()

    def test_offloading(self):
        if not torch.cuda.is_available():
            print("Skipping GPU offload test (No CUDA)")
            return

        # Start on CPU
        self.assertEqual(next(self.model.parameters()).device.type, 'cpu')

        # Load to GPU
        self.mm.load_model_to_gpu(self.model)
        self.assertEqual(next(self.model.parameters()).device.type, 'cuda')

        # Offload
        self.mm.offload_model(self.model)
        self.assertEqual(next(self.model.parameters()).device.type, 'cpu')

    def test_context_manager(self):
        if not torch.cuda.is_available():
            return
            
        with self.mm.execution_scope(self.model, "TestModel"):
            self.assertEqual(next(self.model.parameters()).device.type, 'cuda')
        
        # Should be back on CPU
        self.assertEqual(next(self.model.parameters()).device.type, 'cpu')

if __name__ == '__main__':
    unittest.main()

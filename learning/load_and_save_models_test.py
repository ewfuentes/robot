import unittest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from dataclasses import dataclass
from collections import namedtuple
from typing import Tuple

from learning.load_and_save_models import save_model, load_model

# Define structured types for testing
@dataclass
class ModelInputs:
    features: torch.Tensor
    mask: torch.Tensor

@dataclass
class ModelOutputs:
    logits: torch.Tensor
    attention: torch.Tensor

ModelState = namedtuple('ModelState', ['hidden', 'cell'])

class StructuredInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 3)
    
    def forward(self, inputs: ModelInputs) -> ModelOutputs:
        x = self.linear1(inputs.features)
        x = x * inputs.mask
        attention = torch.softmax(x, dim=-1)
        logits = self.linear2(x)
        return ModelOutputs(logits=logits, attention=attention)

class StatefulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 5, batch_first=True)
        
    def forward(self, x: torch.Tensor, state: ModelState) -> Tuple[torch.Tensor, ModelState]:
        output, (h, c) = self.lstm(x, (state.hidden, state.cell))
        return output, ModelState(hidden=h, cell=c)

class TestStructuredModels(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_dataclass_io_model(self):
        model = StructuredInputModel()
        example_inputs = (ModelInputs(
            features=torch.randn(3, 10),
            mask=torch.ones(3, 5)
        ),)
        
        # Save model
        save_path = self.test_dir / "structured_model"
        save_model(
            model=model,
            save_path=save_path,
            example_model_inputs=example_inputs
        )
        
        # Load and verify
        loaded_model = load_model(save_path)
        original_output = model(*example_inputs)
        loaded_output = loaded_model(*example_inputs)
        
        self.assertTrue(torch.allclose(original_output.logits, loaded_output.logits))
        self.assertTrue(torch.allclose(original_output.attention, loaded_output.attention))
        
    def test_namedtuple_state_model(self):
        model = StatefulModel()
        initial_state = ModelState(
            hidden=torch.randn(1, 3, 5),
            cell=torch.randn(1, 3, 5)
        )
        example_inputs = (
            torch.randn(3, 4, 10),  # batch_size=3, seq_len=4, features=10
            initial_state
        )
        
        # Save model
        save_path = self.test_dir / "stateful_model"
        save_model(
            model=model,
            save_path=save_path,
            example_model_inputs=example_inputs
        )
        
        # Load and verify
        loaded_model = load_model(save_path)
        original_output, original_state = model(*example_inputs)
        loaded_output, loaded_state = loaded_model(*example_inputs)
        
        self.assertTrue(torch.allclose(original_output, loaded_output))
        self.assertTrue(torch.allclose(original_state.hidden, loaded_state.hidden))
        self.assertTrue(torch.allclose(original_state.cell, loaded_state.cell))

if __name__ == "__main__":
    unittest.main()
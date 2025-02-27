import unittest
import common.torch as torch
import common.torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from dataclasses import dataclass
from collections import namedtuple
from typing import Tuple

from learning.load_and_save_models import save_model, load_model

# Simple model for basic tests
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

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

# Models with structured I/O
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

class TestLoadAndSaveModels(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.simple_model = SimpleModel()
        self.simple_input = (torch.randn(3, 10),)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_save_and_load_simple_model(self):
        save_path = self.test_dir / "test_model"
        save_model(
            model=self.simple_model,
            save_path=save_path,
            example_model_inputs=self.simple_input,
            aux_information={"test_info": "test"}
        )
        
        # Verify files exist
        self.assertTrue((save_path / "model_weights.pt").exists())
        self.assertTrue((save_path / "model.pt").exists())
        self.assertTrue((save_path / "aux.json").exists())
        self.assertTrue((save_path / "commit_hash.txt").exists())
        self.assertTrue((save_path / "diff.txt").exists())
        self.assertTrue((save_path / "input_output.tar").exists())
        
        loaded_model = load_model(save_path)
        original_output = self.simple_model(*self.simple_input)
        loaded_output = loaded_model(*self.simple_input)
        self.assertTrue(torch.allclose(original_output, loaded_output))
        
    def test_load_model_different_device(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        save_path = self.test_dir / "test_model_gpu"
        save_model(
            model=self.simple_model,
            save_path=save_path,
            example_model_inputs=self.simple_input
        )
        
        loaded_model = load_model(save_path, device="cuda")
        self.assertEqual(next(loaded_model.parameters()).device.type, "cuda")
        
    def test_save_model_with_aux_info(self):
        save_path = self.test_dir / "test_model_aux"
        aux_info = {
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32
        }
        
        save_model(
            model=self.simple_model,
            save_path=save_path,
            example_model_inputs=self.simple_input,
            aux_information=aux_info
        )
        
        import json
        with open(save_path / "aux.json", "r") as f:
            loaded_aux = json.load(f)
            
        self.assertEqual(loaded_aux["learning_rate"], aux_info["learning_rate"])
        self.assertEqual(loaded_aux["epochs"], aux_info["epochs"])
        self.assertEqual(loaded_aux["batch_size"], aux_info["batch_size"])
        self.assertIn("current_time", loaded_aux)

    def test_dataclass_io_model(self):
        model = StructuredInputModel()
        example_inputs = (ModelInputs(
            features=torch.randn(3, 10),
            mask=torch.ones(3, 5)
        ),)
        
        save_path = self.test_dir / "structured_model"
        save_model(
            model=model,
            save_path=save_path,
            example_model_inputs=example_inputs
        )
        
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
            torch.randn(3, 4, 10),
            initial_state
        )
        
        save_path = self.test_dir / "stateful_model"
        save_model(
            model=model,
            save_path=save_path,
            example_model_inputs=example_inputs
        )
        
        loaded_model = load_model(save_path)
        original_output, original_state = model(*example_inputs)
        loaded_output, loaded_state = loaded_model(*example_inputs)
        
        self.assertTrue(torch.allclose(original_output, loaded_output))
        self.assertTrue(torch.allclose(original_state.hidden, loaded_state.hidden))
        self.assertTrue(torch.allclose(original_state.cell, loaded_state.cell))

if __name__ == "__main__":
    unittest.main()

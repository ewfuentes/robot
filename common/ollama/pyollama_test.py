import unittest

from common.ollama import pyollama


class pyollama_test(unittest.TestCase):
    def test_tiny_model(self):
        # Setup
        with pyollama.Ollama('smollm2:135m') as chat:
            # Action
            response = chat('Why is the sky blue? Please give a one sentence answer.')

            # Verification
            self.assertGreater(len(response), 0)


if __name__ == "__main__":
    unittest.main()

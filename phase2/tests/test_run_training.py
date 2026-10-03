import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

from phase2.run_training import parse_args, main
from phase2.src.config import LMConfig

class TestRunTraining(unittest.TestCase):

    @patch("sys.argv", ["run_training.py", 
                        "--language", "hindi",
                        "--train-bin", "dummy_train.bin",
                        "--val-bin", "dummy_val.bin",
                        "--output-dir", "dummy_out",
                        "--batch-size", "32"])
    def test_parse_args(self):
        args = parse_args()
        self.assertEqual(args.language, "hindi")
        self.assertEqual(args.train_bin, Path("dummy_train.bin"))
        self.assertEqual(args.val_bin, Path("dummy_val.bin"))
        self.assertEqual(args.output_dir, Path("dummy_out"))
        self.assertEqual(args.batch_size, 32)
        self.assertFalse(args.resume)
        
    @patch("sys.argv", ["run_training.py", 
                        "--language", "nepali",
                        "--train-bin", "dummy_train.bin",
                        "--val-bin", "dummy_val.bin",
                        "--output-dir", "dummy_out",
                        "--resume"])
    def test_parse_args_resume(self):
        args = parse_args()
        self.assertEqual(args.language, "nepali")
        self.assertTrue(args.resume)

    @patch("phase2.run_training.trainer.train")
    @patch("phase2.run_training.TokenDataset")
    @patch("phase2.run_training.DecoderOnlyLM")
    @patch("phase2.run_training.parse_args")
    def test_main_execution(self, mock_parse_args, mock_model, mock_dataset, mock_train):
        # Mock args
        args = MagicMock()
        args.language = "hindi"
        args.train_bin = MagicMock()
        args.val_bin = MagicMock()
        args.output_dir = Path("mock_out")
        args.batch_size = 8
        args.grad_accum_steps = 2
        args.learning_rate = 1e-4
        args.max_steps = 100
        args.warmup_steps = 10
        args.eval_interval = 50
        args.checkpoint_interval = 50
        args.seed = 42
        args.resume = False
        
        args.train_bin.exists.return_value = True
        args.val_bin.exists.return_value = True
        
        mock_parse_args.return_value = args
        
        main()
        
        # Verify TokenDataset initialized correctly
        self.assertEqual(mock_dataset.call_count, 2)
        
        # Verify DecoderOnlyLM initialized with default LMConfig
        mock_model.assert_called_once()
        lm_cfg_passed = mock_model.call_args[0][0]
        self.assertEqual(type(lm_cfg_passed).__name__, "LMConfig")
        self.assertEqual(lm_cfg_passed.d_model, 448) # Default architecture
        
        # Verify train is called with correct arguments
        mock_train.assert_called_once()
        train_kwargs = mock_train.call_args[1]
        
        self.assertEqual(train_kwargs["cfg"].language, "hindi")
        self.assertEqual(train_kwargs["cfg"].batch_size, 8)
        self.assertEqual(train_kwargs["cfg"].output_dir, Path("mock_out"))
        self.assertIsNone(train_kwargs["resume_from"])

    @patch("phase2.run_training.trainer.get_latest_checkpoint")
    @patch("phase2.run_training.trainer.train")
    @patch("phase2.run_training.TokenDataset")
    @patch("phase2.run_training.DecoderOnlyLM")
    @patch("phase2.run_training.parse_args")
    def test_main_resume(self, mock_parse_args, mock_model, mock_dataset, mock_train, mock_latest):
        args = MagicMock()
        args.language = "nepali"
        args.train_bin = MagicMock()
        args.val_bin = MagicMock()
        args.train_bin.exists.return_value = True
        args.val_bin.exists.return_value = True
        args.resume = True
        args.output_dir = Path("mock_out")
        args.batch_size = 8
        args.grad_accum_steps = 2
        args.learning_rate = 1e-4
        args.max_steps = 100
        args.warmup_steps = 10
        args.eval_interval = 50
        args.checkpoint_interval = 50
        args.seed = 42
        mock_parse_args.return_value = args
        
        mock_latest.return_value = Path("mock_out/step_100.pt")
        
        main()
        
        train_kwargs = mock_train.call_args[1]
        self.assertEqual(train_kwargs["resume_from"], Path("mock_out/step_100.pt"))

if __name__ == "__main__":
    unittest.main()

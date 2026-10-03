import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import LMConfig
from src.model import InputEmbeddings, CausalSelfAttention, FeedForward, TransformerBlock, DecoderOnlyLM

class TestInputEmbeddings(unittest.TestCase):
    def setUp(self):
        self.config = LMConfig()
        self.model = InputEmbeddings(self.config)
        self.model.eval()

    def test_shape(self):
        B, T = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        output = self.model(input_ids)
        self.assertEqual(output.shape, (B, T, self.config.d_model))

    def test_batch_handling(self):
        B, T = 4, 10
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        output = self.model(input_ids)
        self.assertEqual(output.shape, (B, T, self.config.d_model))

    def test_max_seq_len(self):
        B, T = 1, self.config.max_seq_len
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        try:
            output = self.model(input_ids)
            self.assertEqual(output.shape, (B, T, self.config.d_model))
        except Exception as e:
            self.fail(f"T={self.config.max_seq_len} failed with exception: {e}")

    def test_excessive_seq_len(self):
        B, T = 1, self.config.max_seq_len + 1
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        with self.assertRaisesRegex(ValueError, "exceeds maximum sequence length"):
            self.model(input_ids)

    def test_position_embeddings(self):
        B, T = 1, 16
        input_ids = torch.zeros((B, T), dtype=torch.long)
        output = self.model(input_ids)
        self.assertFalse(torch.allclose(output[0, 0, :], output[0, 1, :]))

    def test_gradient_flow(self):
        self.model.train()
        B, T = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        output = self.model(input_ids)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(self.model.token_embedding.weight.grad)
        self.assertIsNotNone(self.model.position_embedding.weight.grad)
        self.assertTrue(torch.any(self.model.token_embedding.weight.grad != 0))
        self.assertTrue(torch.any(self.model.position_embedding.weight.grad != 0))

    def test_dropout(self):
        B, T = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        self.model.eval()
        out1 = self.model(input_ids)
        out2 = self.model(input_ids)
        self.assertTrue(torch.allclose(out1, out2))
        self.model.train()
        out3 = self.model(input_ids)
        out4 = self.model(input_ids)
        self.assertFalse(torch.allclose(out3, out4))


class TestCausalSelfAttention(unittest.TestCase):
    def setUp(self):
        self.config = LMConfig()
        self.attn = CausalSelfAttention(self.config)
        self.attn.eval()
        
    def test_output_shape(self):
        B, T, C = 2, 16, self.config.d_model
        x = torch.randn(B, T, C)
        out = self.attn(x)
        self.assertEqual(out.shape, (B, T, C))

    def test_head_dimensions(self):
        self.assertEqual(self.attn.n_heads, 7)
        self.assertEqual(self.attn.head_dim, 64)
        
    def test_attention_weight_shape(self):
        B, T, C = 2, 16, self.config.d_model
        x = torch.randn(B, T, C)
        out, weights = self.attn(x, return_attention=True)
        self.assertEqual(weights.shape, (B, self.config.n_heads, T, T))
        
    def test_probability_normalization(self):
        B, T, C = 2, 16, self.config.d_model
        x = torch.randn(B, T, C)
        _, weights = self.attn(x, return_attention=True)
        
        sums = weights.sum(dim=-1)
        expected = torch.ones_like(sums)
        self.assertTrue(torch.allclose(sums, expected, atol=1e-5))

    def test_causal_mask_structure(self):
        B, T, C = 1, 4, self.config.d_model
        x = torch.randn(B, T, C)
        _, weights = self.attn(x, return_attention=True)
        
        for b in range(B):
            for h in range(self.config.n_heads):
                for i in range(T):
                    for j in range(T):
                        if j > i:
                            self.assertAlmostEqual(weights[b, h, i, j].item(), 0.0, places=5)
                
                # Check that valid past/current positions are not all zero for a given query
                for i in range(T):
                    self.assertTrue(torch.any(weights[b, h, i, :i+1] > 1e-5))

    def test_no_future_leakage(self):
        B, T, C = 1, 16, self.config.d_model
        x1 = torch.randn(B, T, C)
        
        x2 = x1.clone()
        x2[:, 8:, :] = torch.randn(B, T - 8, C)
        
        out1 = self.attn(x1)
        out2 = self.attn(x2)
        
        self.assertTrue(torch.allclose(out1[:, :8, :], out2[:, :8, :], atol=1e-5))
        self.assertFalse(torch.allclose(out1[:, 8:, :], out2[:, 8:, :], atol=1e-5))

    def test_gradient_flow(self):
        self.attn.train()
        B, T, C = 2, 16, self.config.d_model
        x = torch.randn(B, T, C, requires_grad=True)
        
        out = self.attn(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(self.attn.q_proj.weight.grad)
        self.assertIsNotNone(self.attn.k_proj.weight.grad)
        self.assertIsNotNone(self.attn.v_proj.weight.grad)
        self.assertIsNotNone(self.attn.out_proj.weight.grad)
        
        self.assertTrue(torch.any(self.attn.q_proj.weight.grad != 0))
        self.assertTrue(torch.any(self.attn.k_proj.weight.grad != 0))
        self.assertTrue(torch.any(self.attn.v_proj.weight.grad != 0))
        self.assertTrue(torch.any(self.attn.out_proj.weight.grad != 0))

    def test_invalid_config(self):
        bad_config = LMConfig(d_model=448, n_heads=9)
        with self.assertRaisesRegex(ValueError, "divisible"):
            CausalSelfAttention(bad_config)

    def test_variable_sequence_length(self):
        for T in [1, 4, 16, 32]:
            B, C = 2, self.config.d_model
            x = torch.randn(B, T, C)
            out = self.attn(x)
            self.assertEqual(out.shape, (B, T, C))

    def test_max_sequence_length(self):
        B, T, C = 1, self.config.max_seq_len, self.config.d_model
        x = torch.randn(B, T, C)
        out = self.attn(x)
        self.assertEqual(out.shape, (B, T, C))

    def test_excessive_sequence_length(self):
        B, T, C = 1, self.config.max_seq_len + 1, self.config.d_model
        x = torch.randn(B, T, C)
        with self.assertRaisesRegex(ValueError, "exceeds maximum sequence length"):
            self.attn(x)


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        self.config = LMConfig()
        self.block = TransformerBlock(self.config)
        self.block.eval()

    def test_output_shape(self):
        B, T, C = 2, 16, self.config.d_model
        x = torch.randn(B, T, C)
        out = self.block(x)
        self.assertEqual(out.shape, (B, T, C))

    def test_residual_is_active(self):
        B, T, C = 2, 16, self.config.d_model
        x = torch.randn(B, T, C)
        out = self.block(x)
        self.assertFalse(torch.allclose(out, torch.zeros_like(out)))
        cos_sim = torch.nn.functional.cosine_similarity(
            out.reshape(-1), x.reshape(-1), dim=0
        )
        self.assertGreater(cos_sim.item(), 0.0)

    def test_two_layernorms(self):
        layernorms = [m for m in self.block.children() if isinstance(m, torch.nn.LayerNorm)]
        self.assertEqual(len(layernorms), 2)

    def test_layernorm_names(self):
        names = [name for name, _ in self.block.named_children()
                 if isinstance(getattr(self.block, name), torch.nn.LayerNorm)]
        self.assertIn('ln_1', names)
        self.assertIn('ln_2', names)

    def test_ffn_dimensions(self):
        ffn = self.block.ffn
        linear_in = ffn.net[0]
        linear_out = ffn.net[3]
        self.assertEqual(linear_in.in_features, self.config.d_model)
        self.assertEqual(linear_in.out_features, self.config.d_ff)
        self.assertEqual(linear_out.in_features, self.config.d_ff)
        self.assertEqual(linear_out.out_features, self.config.d_model)

    def test_gradient_flow(self):
        self.block.train()
        B, T, C = 2, 16, self.config.d_model
        x = torch.randn(B, T, C, requires_grad=True)
        out = self.block(x)
        out.sum().backward()
        self.assertIsNotNone(self.block.ln_1.weight.grad)
        self.assertIsNotNone(self.block.ln_2.weight.grad)
        self.assertIsNotNone(self.block.attn.q_proj.weight.grad)
        self.assertIsNotNone(self.block.attn.k_proj.weight.grad)
        self.assertIsNotNone(self.block.attn.v_proj.weight.grad)
        self.assertIsNotNone(self.block.attn.out_proj.weight.grad)
        self.assertIsNotNone(self.block.ffn.net[0].weight.grad)
        self.assertIsNotNone(self.block.ffn.net[3].weight.grad)
        for p in [self.block.ln_1.weight, self.block.ln_2.weight,
                  self.block.attn.q_proj.weight, self.block.attn.out_proj.weight,
                  self.block.ffn.net[0].weight, self.block.ffn.net[3].weight]:
            self.assertTrue(torch.any(p.grad != 0))

    def test_variable_sequence_length(self):
        for T in [1, 4, 16, 32]:
            x = torch.randn(2, T, self.config.d_model)
            out = self.block(x)
            self.assertEqual(out.shape, (2, T, self.config.d_model))

    def test_max_sequence_length(self):
        B, T, C = 1, self.config.max_seq_len, self.config.d_model
        x = torch.randn(B, T, C)
        out = self.block(x)
        self.assertEqual(out.shape, (B, T, C))

    def test_excessive_sequence_length(self):
        B, T, C = 1, self.config.max_seq_len + 1, self.config.d_model
        x = torch.randn(B, T, C)
        with self.assertRaisesRegex(ValueError, "exceeds maximum sequence length"):
            self.block(x)

    def test_attention_return_path(self):
        B, T, C = 2, 16, self.config.d_model
        x = torch.randn(B, T, C)
        out, att = self.block(x, return_attention=True)
        self.assertEqual(out.shape, (B, T, C))
        self.assertEqual(att.shape, (B, self.config.n_heads, T, T))
        for i in range(T):
            for j in range(T):
                if j > i:
                    self.assertAlmostEqual(att[0, 0, i, j].item(), 0.0, places=5)

    def test_train_eval_dropout(self):
        B, T, C = 2, 16, self.config.d_model
        x = torch.randn(B, T, C)
        self.block.eval()
        out1 = self.block(x)
        out2 = self.block(x)
        self.assertTrue(torch.allclose(out1, out2))
        self.block.train()
        out3 = self.block(x)
        out4 = self.block(x)
        self.assertFalse(torch.allclose(out3, out4))

    def test_parameter_count(self):
        total = sum(p.numel() for p in self.block.parameters())
        self.assertEqual(total, 2_643_904)


class TestDecoderOnlyLM(unittest.TestCase):
    def setUp(self):
        self.config = LMConfig()
        self.model = DecoderOnlyLM(self.config)
        self.model.eval()

    # --- Parameter accounting ---

    def test_total_parameter_count(self):
        total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(total, 25_871_504)

    def test_exactly_8_blocks(self):
        self.assertEqual(len(self.model.blocks), 8)

    def test_vocab_size(self):
        self.assertEqual(self.model.config.vocab_size, 10_000)

    def test_d_model(self):
        self.assertEqual(self.model.config.d_model, 448)

    def test_n_heads(self):
        self.assertEqual(self.model.config.n_heads, 7)

    def test_context_length(self):
        self.assertEqual(self.model.config.max_seq_len, 512)

    def test_weight_tying_identity(self):
        # The lm_head weight and token embedding weight must be the SAME object
        self.assertIs(
            self.model.lm_head.weight,
            self.model.embedding.token_embedding.weight
        )

    def test_weight_tying_shared_storage(self):
        # Verify same underlying storage
        self.assertEqual(
            self.model.lm_head.weight.data_ptr(),
            self.model.embedding.token_embedding.weight.data_ptr()
        )

    def test_weight_tying_gradient_shared(self):
        # Mutating one must be visible in the other
        self.model.train()
        B, T = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        logits = self.model(input_ids)
        loss = logits.sum()
        loss.backward()
        # Both references should yield the exact same grad tensor
        self.assertIs(
            self.model.lm_head.weight.grad,
            self.model.embedding.token_embedding.weight.grad
        )

    def test_lm_head_bias_exists(self):
        self.assertIsNotNone(self.model.lm_head.bias)
        self.assertEqual(self.model.lm_head.bias.numel(), 10_000)

    def test_no_duplicate_vocab_embedding(self):
        # Count how many parameters have vocab_size * d_model shape
        large_vocab_params = [
            p for p in self.model.parameters()
            if p.shape == (self.config.vocab_size, self.config.d_model)
        ]
        # Weight tying means there should be exactly ONE such parameter
        self.assertEqual(len(large_vocab_params), 1)

    def test_all_parameters_trainable(self):
        for name, p in self.model.named_parameters():
            self.assertTrue(p.requires_grad, f"{name} is not trainable")

    # --- Forward pass shapes ---

    def test_forward_shape(self):
        B, T = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        logits = self.model(input_ids)
        self.assertEqual(logits.shape, (B, T, self.config.vocab_size))

    def test_forward_variable_seq_len(self):
        for T in [1, 4, 16, 32]:
            input_ids = torch.randint(0, self.config.vocab_size, (2, T))
            logits = self.model(input_ids)
            self.assertEqual(logits.shape, (2, T, self.config.vocab_size))

    def test_forward_max_seq_len(self):
        B, T = 1, self.config.max_seq_len
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        logits = self.model(input_ids)
        self.assertEqual(logits.shape, (B, T, self.config.vocab_size))

    def test_forward_excessive_seq_len(self):
        B, T = 1, self.config.max_seq_len + 1
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        with self.assertRaises((ValueError, RuntimeError)):
            self.model(input_ids)

    # --- Logit quality ---

    def test_logits_are_finite(self):
        B, T = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        logits = self.model(input_ids)
        self.assertTrue(torch.isfinite(logits).all())

    def test_logits_final_dim(self):
        B, T = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        logits = self.model(input_ids)
        self.assertEqual(logits.shape[-1], 10_000)

    # --- Full gradient flow ---

    def test_full_gradient_flow(self):
        self.model.train()
        B, T = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        logits = self.model(input_ids)
        loss = logits.sum()
        loss.backward()

        # Embedding
        self.assertTrue(torch.any(self.model.embedding.token_embedding.weight.grad != 0))
        self.assertTrue(torch.any(self.model.embedding.position_embedding.weight.grad != 0))

        # Attention (check first and last block)
        for idx in [0, 7]:
            block = self.model.blocks[idx]
            self.assertTrue(torch.any(block.attn.q_proj.weight.grad != 0))
            self.assertTrue(torch.any(block.attn.out_proj.weight.grad != 0))

        # FFN (check first and last block)
        for idx in [0, 7]:
            block = self.model.blocks[idx]
            self.assertTrue(torch.any(block.ffn.net[0].weight.grad != 0))
            self.assertTrue(torch.any(block.ffn.net[3].weight.grad != 0))

        # Final LayerNorm
        self.assertTrue(torch.any(self.model.ln_f.weight.grad != 0))

        # LM head bias
        self.assertIsNotNone(self.model.lm_head.bias.grad)
        self.assertTrue(torch.any(self.model.lm_head.bias.grad != 0))

    # --- Attention return ---

    def test_attention_return(self):
        B, T = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        logits, attentions = self.model(input_ids, return_attention=True)

        self.assertEqual(logits.shape, (B, T, self.config.vocab_size))
        self.assertEqual(len(attentions), 8)

        for layer_att in attentions:
            self.assertEqual(layer_att.shape, (B, self.config.n_heads, T, T))
            # Verify causal masking
            for i in range(T):
                for j in range(T):
                    if j > i:
                        self.assertAlmostEqual(
                            layer_att[0, 0, i, j].item(), 0.0, places=5
                        )

    def test_normal_forward_no_attention_stored(self):
        B, T = 2, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, T))
        result = self.model(input_ids)
        # Normal forward returns only logits, not a tuple
        self.assertIsInstance(result, torch.Tensor)


if __name__ == '__main__':
    unittest.main()

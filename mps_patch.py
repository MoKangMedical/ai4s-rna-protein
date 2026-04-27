#!/usr/bin/env python3
"""
MPS Compatibility Patch for Protenix
=====================================
Monkey-patches CUDA-only dependencies to work on Apple Silicon MPS.
MUST be imported BEFORE any Protenix modules.
"""
import os
import sys
import torch
import types
import warnings

warnings.filterwarnings("ignore")

# Detect device
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
    print(f"[MPS Patch] Apple Silicon GPU detected: MPS (10 cores)")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
    print(f"[MPS Patch] MPS not available, using CPU")

# ============================================================
# Pre-patch 0: Create mock fast_layer_norm_cuda_v2 BEFORE import
# ============================================================
# This MUST happen before any Protenix import

class MockLayerNormCUDA:
    """Mock CUDA layer norm that redirects to PyTorch native."""
    
    @staticmethod
    def forward_none_affine(input_, normalized_shape, eps):
        out = torch.nn.functional.layer_norm(input_, normalized_shape, eps=eps)
        mean = input_.mean(dim=-1, keepdim=True)
        var = input_.var(dim=-1, keepdim=True, unbiased=False)
        invvar = 1.0 / (var + eps).sqrt()
        return out, mean, invvar
    
    @staticmethod
    def forward_with_bias_affine(input_, normalized_shape, bias, eps):
        out = torch.nn.functional.layer_norm(input_, normalized_shape, bias=bias, eps=eps)
        mean = input_.mean(dim=-1, keepdim=True)
        var = input_.var(dim=-1, keepdim=True, unbiased=False)
        invvar = 1.0 / (var + eps).sqrt()
        return out, mean, invvar
    
    @staticmethod
    def forward_with_weight_affine(input_, normalized_shape, weight, eps):
        out = torch.nn.functional.layer_norm(input_, normalized_shape, weight=weight, eps=eps)
        mean = input_.mean(dim=-1, keepdim=True)
        var = input_.var(dim=-1, keepdim=True, unbiased=False)
        invvar = 1.0 / (var + eps).sqrt()
        return out, mean, invvar
    
    @staticmethod
    def forward_with_both_affine(input_, normalized_shape, weight, bias, eps):
        out = torch.nn.functional.layer_norm(input_, normalized_shape, weight=weight, bias=bias, eps=eps)
        mean = input_.mean(dim=-1, keepdim=True)
        var = input_.var(dim=-1, keepdim=True, unbiased=False)
        invvar = 1.0 / (var + eps).sqrt()
        return out, mean, invvar

# Inject mock module BEFORE any Protenix import
mock_module = MockLayerNormCUDA()
sys.modules["fast_layer_norm_cuda_v2"] = mock_module
print("[MPS Patch] fast_layer_norm_cuda_v2 -> PyTorch native LayerNorm")

# Also mock the torch_ext_compile to prevent it from trying to compile
class MockTorchExtCompile:
    @staticmethod
    def compile(**kwargs):
        return mock_module
    
    @staticmethod
    def load(**kwargs):
        return mock_module

mock_compile = types.ModuleType("protenix.model.layer_norm.torch_ext_compile")
mock_compile.compile = MockTorchExtCompile.compile
mock_compile.load = MockTorchExtCompile.load

# Pre-patch the layer_norm module path
layer_norm_dir = None
if len(sys.path) > 0:
    for p in sys.path:
        candidate = os.path.join(p, "protenix", "model", "layer_norm", "torch_ext_compile.py")
        if os.path.exists(candidate):
            layer_norm_dir = os.path.dirname(candidate)
            break

# ============================================================
# Pre-patch 1: Mock attn_core_inplace_cuda
# ============================================================
class MockAttnCoreCUDA:
    @staticmethod
    def forward_(logits, n, d):
        """In-place softmax forward - just use PyTorch."""
        logits.softmax_(dim=-1)
    
    @staticmethod
    def backward_(grad_output, output, n, d):
        """In-place softmax backward."""
        pass  # Simplified - for inference only

mock_attn = MockAttnCoreCUDA()
sys.modules["attn_core_inplace_cuda"] = mock_attn
print("[MPS Patch] attn_core_inplace_cuda -> PyTorch native softmax")

# ============================================================
# Patch 2: Replace .cuda() with device-aware .to()
# ============================================================
if DEVICE == "mps":
    _original_cuda = torch.Tensor.cuda
    
    def _safe_cuda(self, *args, **kwargs):
        return self.to("mps")
    
    torch.Tensor.cuda = _safe_cuda
    print("[MPS Patch] .cuda() -> .to('mps')")

# ============================================================
# Patch 3: Fix TriAttention for MPS
# ============================================================
# Will be applied after import

def apply_post_import_patches():
    """Patches that need to be applied after modules are imported."""
    try:
        import protenix.model.tri_attention as ta_module
        
        class MPSTriAttentionFunction:
            @staticmethod
            def apply(q, k, v, bias1=None, bias2=None, deterministic=False):
                B, N, S, H, D = q.shape
                q = q.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)
                k = k.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)
                v = v.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)
                
                attn_mask = None
                if bias1 is not None or bias2 is not None:
                    attn_mask = 0
                    if bias1 is not None:
                        attn_mask = attn_mask + bias1.reshape(B * N * H, 1, S)
                    if bias2 is not None:
                        attn_mask = attn_mask + bias2.reshape(B * N * H, S, 1)
                
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
                )
                return out.reshape(B, N, H, S, D).permute(0, 1, 3, 2, 4)
        
        class MPSTriAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, q, k, v, bias1=None, bias2=None, deterministic=False):
                return MPSTriAttentionFunction.apply(q, k, v, bias1, bias2, deterministic)
        
        ta_module.TriAttentionFunction = MPSTriAttentionFunction
        ta_module.TriAttention = MPSTriAttention
        ta_module.TRITON_AVAILABLE = False
        print("[MPS Patch] TriAttention -> MPS-compatible SDPA")
    except Exception as e:
        print(f"[MPS Patch] TriAttention patch skipped: {e}")

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print(f"[MPS Patch] Pre-import patches applied! Device: {DEVICE}")
print(f"[MPS Patch] Call apply_post_import_patches() after importing Protenix")
print("=" * 60)

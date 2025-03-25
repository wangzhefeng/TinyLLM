import torch
import torch.nn as nn
import pytest
from layers.rms_norm import RMSNorm

def test_rms_norm_initialization():
    # Test valid initialization
    rms_norm = RMSNorm(emb_dim=64)
    assert rms_norm.emb_dim == 64
    assert rms_norm.eps == 1e-5
    assert rms_norm.weight.shape == (64,)
    
    # Test invalid initialization
    with pytest.raises(ValueError):
        RMSNorm(emb_dim=0)
    with pytest.raises(ValueError):
        RMSNorm(emb_dim=64, eps=0)

def test_rms_norm_forward():
    # Setup
    torch.manual_seed(123)
    emb_dim = 8
    batch_size = 2
    seq_len = 3
    
    # Create test input
    x = torch.randn(batch_size, seq_len, emb_dim)
    
    # Initialize layers
    rms_norm = RMSNorm(emb_dim=emb_dim)
    rms_norm_pytorch = nn.RMSNorm(emb_dim, eps=1e-5)
    
    # Test forward pass
    output = rms_norm(x)
    expected_output = rms_norm_pytorch(x)
    
    # Verify results
    assert output.shape == x.shape
    assert torch.allclose(output, expected_output, rtol=1e-5)
    
    # Test invalid input dimension
    with pytest.raises(ValueError):
        rms_norm(torch.randn(batch_size, seq_len, emb_dim + 1))

def test_rms_norm_gradients():
    # Setup
    emb_dim = 16
    x = torch.randn(4, 5, emb_dim, requires_grad=True)
    rms_norm = RMSNorm(emb_dim=emb_dim)
    
    # Forward pass
    output = rms_norm(x)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Verify gradients
    assert x.grad is not None
    assert rms_norm.weight.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(rms_norm.weight.grad).any()

def test_rms_norm_different_eps():
    # Test different epsilon values
    emb_dim = 32
    x = torch.randn(2, 3, emb_dim)
    
    for eps in [1e-5, 1e-6, 1e-4]:
        rms_norm = RMSNorm(emb_dim=emb_dim, eps=eps)
        output = rms_norm(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

if __name__ == "__main__":
    pytest.main()
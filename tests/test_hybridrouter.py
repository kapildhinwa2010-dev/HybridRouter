import pytest
import torch
from hybridrouter import HybridRouter, create_hybridrouter

class TestHybridRouter:
    def test_init(self):
        m = HybridRouter(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = HybridRouter(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_hybridrouter()
        assert isinstance(m, HybridRouter)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

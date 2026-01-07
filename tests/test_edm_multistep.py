"""
Unit tests for multi-step EDM denoising implementation.

Run with: pytest tests/test_edm_multistep.py -v
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.edm.edm_unified_model_multistep import (
    get_denoising_sigmas,
)
from main.edm.sample_edm_multistep import (
    get_sigmas_karras,
    sample_multistep_cfg,
    sample_multistep_deterministic
)


class TestKarrasSigmaSchedule:
    """Tests for Karras sigma schedule generation."""

    def test_sigma_schedule_length(self):
        """Sigma schedule should have correct number of steps."""
        for n in [1, 5, 10, 20]:
            sigmas = get_denoising_sigmas(n, sigma_max=80.0, sigma_min=0.002)
            assert len(sigmas) == n, f"Expected {n} sigmas, got {len(sigmas)}"

    def test_sigma_schedule_descending(self):
        """Sigmas should be in descending order (large to small)."""
        sigmas = get_denoising_sigmas(10, sigma_max=80.0, sigma_min=0.002)
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1], \
                f"Sigmas should be descending: sigma[{i}]={sigmas[i]} vs sigma[{i+1}]={sigmas[i+1]}"

    def test_sigma_schedule_bounds(self):
        """First and last sigma should match max and min."""
        sigma_max = 80.0
        sigma_min = 0.002
        sigmas = get_denoising_sigmas(10, sigma_max=sigma_max, sigma_min=sigma_min)
        assert sigmas[0] == pytest.approx(sigma_max, rel=1e-5), \
            f"First sigma should be {sigma_max}, got {sigmas[0]}"
        assert sigmas[-1] == pytest.approx(sigma_min, rel=1e-5), \
            f"Last sigma should be {sigma_min}, got {sigmas[-1]}"

    def test_sigma_schedule_positive(self):
        """All sigmas should be positive."""
        sigmas = get_denoising_sigmas(10, sigma_max=80.0, sigma_min=0.002)
        assert (sigmas > 0).all(), "All sigmas should be positive"

    def test_different_rho_values(self):
        """Different rho values should produce different schedules."""
        sigmas_rho5 = get_denoising_sigmas(10, sigma_max=80.0, sigma_min=0.002, rho=5.0)
        sigmas_rho7 = get_denoising_sigmas(10, sigma_max=80.0, sigma_min=0.002, rho=7.0)
        sigmas_rho9 = get_denoising_sigmas(10, sigma_max=80.0, sigma_min=0.002, rho=9.0)

        # First and last should be same (bounds) - use approx for floating point
        assert sigmas_rho5[0].item() == pytest.approx(sigmas_rho7[0].item(), rel=1e-5)
        assert sigmas_rho7[0].item() == pytest.approx(sigmas_rho9[0].item(), rel=1e-5)
        assert sigmas_rho5[-1].item() == pytest.approx(sigmas_rho7[-1].item(), rel=1e-5)
        assert sigmas_rho7[-1].item() == pytest.approx(sigmas_rho9[-1].item(), rel=1e-5)

        # Middle values should differ
        assert not torch.allclose(sigmas_rho5, sigmas_rho7)
        assert not torch.allclose(sigmas_rho7, sigmas_rho9)


class TestSigmaScheduleConsistency:
    """Test consistency between different sigma generation functions."""

    def test_both_functions_match(self):
        """get_denoising_sigmas and get_sigmas_karras should produce same results."""
        sigmas1 = get_denoising_sigmas(10, sigma_max=80.0, sigma_min=0.002, rho=7.0)
        sigmas2 = get_sigmas_karras(10, sigma_min=0.002, sigma_max=80.0, rho=7.0)
        assert torch.allclose(sigmas1, sigmas2), \
            "Both sigma generation functions should produce identical results"


class MockGenerator:
    """Mock generator for testing sampling functions."""

    def __init__(self, resolution=64, return_input=False):
        self.resolution = resolution
        self.return_input = return_input
        self.call_count = 0

    def __call__(self, x, sigma, labels):
        self.call_count += 1
        if self.return_input:
            # Return input scaled down (simple denoising mock)
            return x * 0.9
        else:
            # Return random image (not realistic but tests shape)
            return torch.randn_like(x)


class TestMultistepSampling:
    """Tests for multi-step sampling functions."""

    def test_sample_multistep_output_shape(self):
        """Output should have correct shape."""
        generator = MockGenerator()
        batch_size = 4
        resolution = 64
        num_classes = 1000

        noise = torch.randn(batch_size, 3, resolution, resolution)
        labels = torch.eye(num_classes)[:batch_size]

        output = sample_multistep_cfg(
            generator=generator,
            noise=noise,
            labels=labels,
            num_steps=10,
            sigma_max=80.0,
            sigma_min=0.002
        )

        assert output.shape == (batch_size, 3, resolution, resolution), \
            f"Expected shape {(batch_size, 3, resolution, resolution)}, got {output.shape}"

    def test_sample_multistep_calls_generator_correct_times(self):
        """Generator should be called once per denoising step."""
        generator = MockGenerator()
        noise = torch.randn(2, 3, 64, 64)
        labels = torch.eye(1000)[:2]

        num_steps = 10
        sample_multistep_cfg(
            generator=generator,
            noise=noise,
            labels=labels,
            num_steps=num_steps,
            guidance_scale=1.0  # Disable CFG for single call per step
        )

        assert generator.call_count == num_steps, \
            f"Generator should be called {num_steps} times, was called {generator.call_count} times"

    def test_deterministic_sampling_reproducible(self):
        """Deterministic sampling should be reproducible with same seed."""
        generator = MockGenerator(return_input=True)
        labels = torch.eye(1000)[:2]

        torch.manual_seed(42)
        noise1 = torch.randn(2, 3, 64, 64)
        output1 = sample_multistep_deterministic(
            generator=generator,
            noise=noise1,
            labels=labels,
            num_steps=5
        )

        # Reset generator call count
        generator.call_count = 0

        torch.manual_seed(42)
        noise2 = torch.randn(2, 3, 64, 64)
        output2 = sample_multistep_deterministic(
            generator=generator,
            noise=noise2,
            labels=labels,
            num_steps=5
        )

        assert torch.allclose(output1, output2), \
            "Deterministic sampling should produce same output with same seed"


class TestCFGSampling:
    """Tests for classifier-free guidance sampling."""

    def test_cfg_calls_generator_twice_per_step(self):
        """With CFG > 1, generator should be called twice per step."""
        generator = MockGenerator()
        noise = torch.randn(2, 3, 64, 64)
        labels = torch.eye(1000)[:2]

        num_steps = 5
        sample_multistep_cfg(
            generator=generator,
            noise=noise,
            labels=labels,
            num_steps=num_steps,
            guidance_scale=2.0  # CFG enabled
        )

        expected_calls = num_steps * 2  # Conditional + unconditional per step
        assert generator.call_count == expected_calls, \
            f"With CFG, generator should be called {expected_calls} times, was called {generator.call_count} times"

    def test_cfg_disabled_calls_generator_once_per_step(self):
        """With CFG = 1.0, generator should be called once per step."""
        generator = MockGenerator()
        noise = torch.randn(2, 3, 64, 64)
        labels = torch.eye(1000)[:2]

        num_steps = 5
        sample_multistep_cfg(
            generator=generator,
            noise=noise,
            labels=labels,
            num_steps=num_steps,
            guidance_scale=1.0  # CFG disabled
        )

        assert generator.call_count == num_steps, \
            f"Without CFG, generator should be called {num_steps} times, was called {generator.call_count} times"

    def test_cfg_output_shape(self):
        """CFG sampling should produce correct output shape."""
        generator = MockGenerator()
        batch_size = 4
        noise = torch.randn(batch_size, 3, 64, 64)
        labels = torch.eye(1000)[:batch_size]

        output = sample_multistep_cfg(
            generator=generator,
            noise=noise,
            labels=labels,
            num_steps=10,
            guidance_scale=1.5
        )

        assert output.shape == (batch_size, 3, 64, 64)


class TestEdgesCases:
    """Test edge cases and boundary conditions."""

    def test_single_step(self):
        """Single step sampling should work."""
        generator = MockGenerator()
        noise = torch.randn(2, 3, 64, 64)
        labels = torch.eye(1000)[:2]

        output = sample_multistep_cfg(
            generator=generator,
            noise=noise,
            labels=labels,
            num_steps=1,
            guidance_scale=1.0  # Disable CFG for single call per step
        )

        assert output.shape == (2, 3, 64, 64)
        assert generator.call_count == 1

    def test_batch_size_one(self):
        """Batch size of 1 should work."""
        generator = MockGenerator()
        noise = torch.randn(1, 3, 64, 64)
        labels = torch.eye(1000)[:1]

        output = sample_multistep_cfg(
            generator=generator,
            noise=noise,
            labels=labels,
            num_steps=5
        )

        assert output.shape == (1, 3, 64, 64)

    def test_large_batch(self):
        """Large batch should work."""
        generator = MockGenerator()
        batch_size = 128
        noise = torch.randn(batch_size, 3, 64, 64)
        labels = torch.eye(1000)[:batch_size]

        output = sample_multistep_cfg(
            generator=generator,
            noise=noise,
            labels=labels,
            num_steps=3
        )

        assert output.shape == (batch_size, 3, 64, 64)


class TestDeviceCompatibility:
    """Test device compatibility (CPU by default, CUDA if available)."""

    def test_cpu_sampling(self):
        """Sampling should work on CPU."""
        generator = MockGenerator()
        noise = torch.randn(2, 3, 64, 64)
        labels = torch.eye(1000)[:2]

        output = sample_multistep_cfg(
            generator=generator,
            noise=noise,
            labels=labels,
            num_steps=3
        )

        assert output.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_sigma_generation(self):
        """Sigma generation should work on CUDA."""
        sigmas = get_denoising_sigmas(10, sigma_max=80.0, sigma_min=0.002).cuda()
        assert sigmas.device.type == 'cuda'
        assert len(sigmas) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

import pytest
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import jit
import librosa
import numpy as np
import jax.numpy as jnp
from jax_audio.utils import audio_utils

test_cases = [
    np.random.rand(i).astype(np.float32) for i in [2048, 10 ** 4, 10 ** 6, 10 ** 7]
]
test_cases_jax = [jnp.array(i) for i in test_cases]

test_cases_batch = [
    np.random.rand(i).astype(np.float32) for i in [2048, 10 ** 4, 10 ** 6, 10 ** 7]
]
test_cases_batch_jax = [jnp.array(i) for i in test_cases_batch]


@pytest.mark.parametrize("x", test_cases)
def test_librosa_stft(benchmark, x):

    librosa.stft(np.zeros(x.shape, dtype=x.dtype))
    benchmark(librosa.stft, x)


@pytest.mark.parametrize("x", test_cases_jax)
def test_jax_stft_jitted(benchmark, x):

    jitted = jit(audio_utils.stft)
    jitted(jnp.zeros(x.shape, dtype=x.dtype))  # warm start

    benchmark(jitted, x)


@pytest.mark.parametrize("x", test_cases_jax)
def test_jax_stft(benchmark, x):

    audio_utils.stft(jnp.zeros(x.shape, dtype=x.dtype))  # warm start
    benchmark(audio_utils.stft, x)

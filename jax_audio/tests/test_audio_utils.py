import pytest
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from jax_audio.utils import audio_utils
import librosa
import numpy as np
import jax.numpy as jnp


@pytest.mark.parametrize("x", [2048, 2049, 10 ** 4, 10 ** 6])
def test_stft(x):
    data = np.random.rand(x)
    out_jax = audio_utils.stft(jnp.array(data))

    out_lib = librosa.stft(data, 1024, 256, 1024)

    mag_lib = np.abs(out_lib)
    mag_jax = np.abs(out_jax)

    phase_lib = np.angle(out_lib)
    phase_jax = np.angle(out_jax)

    amplitude_lib = librosa.amplitude_to_db(np.abs(out_lib))
    amplitude_jax = audio_utils.amplitude_to_db(jnp.abs(out_jax))

    # Check if magnitude diff is =< 0.5 (can different diff in < 1% of cases)
    # diff can be caused by float points operations librosa is using float64 jax is using float32
    # on bigger arrays diff will be larger
    assert np.isclose(mag_lib, mag_jax, atol=0.5).sum() > int(
        len(out_lib.flatten()) * 0.99
    )

    # Check if phase diff is =< 0.5 (can be different in < 10% of cases)
    # difference can be caused by float points operations
    # librosa is using complex128 and float64 jax is using complex64 and float32
    assert np.isclose(phase_lib, phase_jax, atol=0.5).sum() > int(
        len(out_lib.flatten()) * 0.9
    )

    # Check if max db diff is =< 0.5dB (can be different in < 0.5% of cases)
    assert np.isclose(amplitude_lib, np.array(amplitude_jax), atol=0.5).sum() > int(
        len(amplitude_lib.flatten()) * 0.995
    )

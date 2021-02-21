from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import List
from typing import Tuple
from typing import Union

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap
from librosa.filters import mel as librosa_mel_fn

from jax_audio.utils.audio_utils import read_wav
from jax_audio.utils.audio_utils import stft


@dataclass
class MelGenBasic(ABC):
    filter_length: int = 2048
    hop_length: int = 256
    win_length: int = 1024
    window: str = "hanning"
    sampling_rate: int = 22050
    n_mel_channels: int = 80
    mel_f_min: float = 0.0
    mel_f_max: float = 8000.0
    max_wav_value = 32768.0
    use_vmap: bool = True

    use_energy: bool = True

    def __post_init__(self):
        self.mel_basis = jnp.asarray(
            librosa_mel_fn(
                self.sampling_rate,
                self.filter_length,
                self.n_mel_channels,
                self.mel_f_min,
                self.mel_f_max,
            )
        )
        # JIT Will compile versions for every diff sized arrays
        # run dataset preprocessing before to create bucket of similar len or compilation times destroy you :D!
        if self.use_vmap:
            self.stft_parser = partial(
                jit(vmap(stft, in_axes=(0, None, None, None)), static_argnums=(1, 2, 3))
            )
        else:
            self.stft_parser = partial(jit(stft, static_argnums=(1, 2, 3)))

    @staticmethod
    def pad_generator(x: List[np.ndarray], max_len: Union[int, None] = None):
        """
        Method used to fast generate padded batch of data to parse later via jitted stft function

        Args:
            x: list of numpy arrays with loaded audio files
            max_len: maximum len of value to pad

        Returns:

        """
        shapes = [i.shape[0] for i in x]
        max_shape = max(shapes)
        if max_len:
            assert max_len >= max_shape
            max_shape = max_len
        new_array = np.zeros((len(x), max_shape))
        for i in range(len(x)):
            new_array[i, : shapes[i]] = x[i]
        return jnp.asarray(new_array), shapes

    def calc_energy(self, magnitudes: jnp.ndarray):
        return jnp.sqrt(jnp.sum(magnitudes ** 2, axis=1 if self.use_vmap else 0))

    @abstractmethod
    def mel_gen(self, audio: jnp.ndarray):
        raise NotImplementedError


@dataclass
class MelGeneratorNV(MelGenBasic):  # Mimic nvidia mel generator for waveglow
    def audio_file_to_mel(self, f_name: str):
        audio, sampling_rate = read_wav(f_name, normalize=False)
        audio_norm = audio / self.max_wav_value
        if self.use_vmap:
            audio_norm = audio_norm.reshape(1, -1)
        return self.mel_gen(audio_norm)

    def audio_file_to_mel_batch(
        self, f_names: List[str], max_len: Union[int, None] = None
    ):
        audio_list = []
        for f_name in f_names:
            audio, sampling_rate = read_wav(f_name, normalize=False, return_jax=False)
            audio_norm = audio / self.max_wav_value
            audio_list.append(audio_norm)
        return self.mel_gen(self.pad_generator(audio_list, max_len)[0])

    def mel_gen(
        self, y: jnp.ndarray, safe_check: bool = True
    ) -> Tuple[jnp.ndarray, Union[jnp.ndarray, None]]:
        """
        Args:
            y: jax numpy ndarray with shape (Batch, T) normalized into [-1, 1] range
            safe_check: bool if true there will be check if all values in array are scaled to >= -1 and <= 1

        Returns: mel -> jax numpy ndarray (Batch, n_mel_channels, R), energy -> jax numpy ndarray (batch, 1, R) or None
        """
        if safe_check:
            assert jnp.min(y) >= -1
            assert jnp.max(y) <= 1

        magnitudes = jnp.abs(
            self.stft_parser(y, self.filter_length, self.hop_length, self.win_length)
        )

        energy = None
        if self.use_energy:
            energy = self.calc_energy(magnitudes ** 1)

        mel_output = jnp.matmul(self.mel_basis, magnitudes)
        return self.spectral_normalization(mel_output), energy

    @staticmethod
    def spectral_normalization(magnitudes: jnp.ndarray):
        return jnp.log(
            jnp.clip(magnitudes, a_min=1e-5, a_max=None) * 1.0
        )  # mimic nvidia version

    @staticmethod
    def spectral_de_normalization(magnitudes: jnp.ndarray):
        return jnp.exp(magnitudes) / 1.0  # mimic nvidia version


@dataclass
class MelGeneratorTF(MelGenBasic):  # Mimic TensorflowTTS version :P
    def audio_file_to_mel(self, f_name: str):
        audio, sampling_rate = read_wav(f_name, normalize=False)
        audio_norm = audio / self.max_wav_value
        if self.use_vmap:
            audio_norm = audio_norm.reshape(1, -1)
        return self.mel_gen(audio_norm)

    def audio_file_to_mel_batch(
        self, f_names: List[str], max_len: Union[int, None] = None
    ):
        audio_list = []
        for f_name in f_names:
            audio, sampling_rate = read_wav(f_name, normalize=False, return_jax=False)
            audio_norm = audio / self.max_wav_value
            audio_list.append(audio_norm)
        return self.mel_gen(self.pad_generator(audio_list, max_len)[0])

    def mel_gen(self, y: jnp.ndarray) -> Tuple[jnp.ndarray, Union[jnp.ndarray, None]]:
        """
        Args:
            y: jax numpy ndarray with shape (Batch, T) normalized into [-1, 1] range
            safe_check: bool if true there will be check if all values in array are scaled to >= -1 and <= 1

        Returns: mel -> jax numpy ndarray (Batch, n_mel_channels, R), energy -> jax numpy ndarray (batch, 1, R) or None
        """
        assert jnp.min(y) >= -1
        assert jnp.max(y) <= 1

        magnitudes = jnp.abs(
            self.stft_parser(y, self.filter_length, self.hop_length, self.win_length)
        )

        energy = None
        if self.use_energy:
            energy = self.calc_energy(magnitudes ** 1)

        return (
            jnp.log10(jnp.maximum(jnp.matmul(self.mel_basis, magnitudes), 1e-10)),
            energy,
        )

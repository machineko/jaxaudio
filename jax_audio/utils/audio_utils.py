import warnings
from typing import Tuple, Union
import jax.numpy as jnp
import numpy as np
import wave
from jax import lax
from jax_audio.utils.audio_exception import ParameterError
from functools import partial
import jax


def get_window(win_name: str, win_length: int, **kwargs) -> jnp.ndarray:
    """
    Args:
        win_name: name of the window function (must exist in jax repository)
        win_length: len of the window
        **kwargs: optional extra arguments for function

    Returns:
        jnp.ndarray -> The window returned by window function
    """
    win_func = getattr(jnp, win_name, None)  # get function call if exists
    if win_func:
        return win_func(win_length, **kwargs)
    else:
        raise NotImplemented(f"{win_name} isn't yet implemented in jax")


def pad_center(data: jnp.ndarray, size: int, axis=-1, **kwargs) -> jnp.ndarray:
    """
    It is copy of librosa pad_center transformed to work with jax
    Warning this is pure function
    https://github.com/librosa/librosa


    Wrapper for jnp.pad to automatically center an array prior to padding.
    >>> # Generate a vector
    >>> data = jnp.ones(5)
    >>> pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

    >>> # Pad a matrix along its first dimension
    >>> data = jnp.ones((3, 5))
    >>> pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : jnp.ndarray
        Vector to be padded and centered

    size : int >= len(data) [scalar]
        Length to pad `data`

    axis : int
        Axis along which to pad and center the data

    kwargs : additional keyword arguments
      arguments passed to `jnp.pad()`

    Returns
    -------
    data_padded : jnp.ndarray
        `data` centered and padded to length `size` along the
        specified axis
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))
    if lpad < 0:
        raise BaseException(f"Target size ({size}) must be at least input size {n}")

    return jnp.pad(data, lengths, **kwargs)


def read_wav(
    f_name: str, return_jax: bool = True, normalize: bool = True, dtype: str = "float32"
) -> Tuple[Union[jnp.ndarray, np.ndarray], float]:
    """

    Args:
        f_name: wav file name
        normalize: bool if true audio get normalized to [0, 1]
        dtype: array dtype
        return_jax: if true return audio as jax array

    Returns: parsed wav file and

    """

    wav_file = wave.open(f_name)

    if wav_file.getnchannels() != 1:
        raise NotImplemented("Reading not mono file isn't implemented yet")

    samples = wav_file.getnframes()
    audio = wav_file.readframes(samples)
    sample_rate = wav_file.getframerate()
    audio = np.frombuffer(audio, dtype=np.int16).astype(dtype)
    if normalize:
        audio = audio / 2 ** 15
    return jnp.asarray(audio) if return_jax else audio, sample_rate


def write_wav(f_name: str, sample_rate: int, audio: np.ndarray, dtype: str = np.int16):
    with wave.open(f_name, "wb") as wav_file:
        params = [1, 2, sample_rate, audio.size, "NONE", "not compressed"]
        wav_file.setparams(params)
        wav_file.writeframes(np.array((audio * (2 ** 15))).astype(dtype))


def stft(
    y: jnp.ndarray,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    window="hanning",
    pad_mode="reflect",
) -> jnp.ndarray:
    """
        Short-time Fourier transform (STFT). based of librosa stft version works with jax numpy arrays
        (both gpu and cpu)

        Warning This is pure function and this function does not split array using strides so without
        jit or gpu usage it can be a lot slower

        original repo -> https://github.com/librosa/librosa

    Args:
        y: input signal

        n_fft:
        The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
        The default value, ``n_fft=2048`` samples, corresponds to a physical
        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
        default sample rate in librosa. This value is well adapted for music
        signals. However, in speech processing, the recommended value is 512,
        corresponding to 23 milliseconds at a sample rate of 22050 Hz.
        In any case, we recommend setting ``n_fft`` to a power of two for
        optimizing the speed of the fast Fourier transform (FFT) algorithm.

        hop_length: number of audio samples between adjacent STFT columns.

        win_length: Each frame of audio is windowed by ``window`` of length ``win_length``

        window: string a jax.numpy supported windowing method

        center:  If ``True``, the signal ``y`` is padded so that frame
        ``D[:, t]`` is centered at ``y[t * hop_length]``.

        pad_mode: jax.numpy supported padding function

    Returns: jnp.ndarray [shape=(1 + n_fft/2, n_frames), dtype=complex64]
        Complex-valued matrix of short-term Fourier transform
        coefficients.

    """

    if not win_length:
        win_length = n_fft

    if not hop_length:
        hop_length = int(win_length // 4)

    # get window function
    fft_window = get_window(window, win_length)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    new_y = jnp.pad(y, int(n_fft // 2), mode=pad_mode)

    n_frames = 1 + (new_y.shape[-1] - win_length) // hop_length

    y_frames = new_y[
        jnp.arange(n_fft)[:, None] + hop_length * jnp.arange(n_frames)[None, :]
    ]

    return jnp.fft.rfft(fft_window * y_frames, axis=0)


def invers_stft_slow(
    stft_matrix, hop_length=256, win_length=1024, window="hanning", dtype=jnp.float32
):
    n_fft = 2 * (stft_matrix.shape[0] - 1)
    fft_window = get_window(window, win_length)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    sq_window = fft_window ** 2

    n_frames = stft_matrix.shape[1]

    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    tmp_irfft = jnp.expand_dims(fft_window, -1) * jnp.fft.irfft(stft_matrix, axis=0)
    y = jnp.zeros(expected_signal_len, dtype=dtype)
    win_sum = jnp.zeros(expected_signal_len, dtype=dtype)
    tmp_n_fft = tmp_irfft.shape[0]

    for frame in range(tmp_irfft.shape[1]):
        sample = frame * hop_length
        y = lax.dynamic_update_slice(
            y, y[sample : (sample + tmp_n_fft)] + tmp_irfft[:, frame], [sample]
        )
        win_sum = lax.dynamic_update_slice(
            win_sum, win_sum[sample : (sample + tmp_n_fft)] + sq_window, [sample]
        )
    # non_zero = win_sum != 0
    y = y / win_sum
    return y, n_fft


def inverse_stft_gpu(
    stft_matrix, hop_length=256, win_length=1024, window="hanning", dtype=np.float32
):

    from jax_audio.utils.cuda_utils import gpu_add_irfft, gpu_add_win
    from numba import cuda

    n_fft = 2 * (stft_matrix.shape[0] - 1)
    n_frames = stft_matrix.shape[1]

    fft_window = get_window(window, win_length)
    fft_window = pad_center(fft_window, n_fft)
    sq_window = fft_window ** 2

    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    tmp_irfft = jnp.expand_dims(fft_window, -1) * jnp.fft.irfft(stft_matrix, axis=0)

    y = cuda.to_device(np.zeros(expected_signal_len, dtype=dtype))

    # Very basic numba cuda kernal is used cause of jax Omnistaging break jax jitted version (which was a lot faster)
    gpu_add_irfft[(32, 8), (8, 8)](y, tmp_irfft, hop_length)

    win_sum = cuda.to_device(np.zeros(expected_signal_len, dtype=dtype))

    gpu_add_win[(32, 8), (8, 8)](win_sum, sq_window, tmp_irfft.shape[1], hop_length)

    y, win_sum = np.asarray(y), np.asarray(win_sum)
    approx_nonzero_indices = win_sum != 0.0
    y[approx_nonzero_indices] /= win_sum[approx_nonzero_indices]
    return y[int(n_fft // 2) : -int(n_fft // 2)]


def create_mask_from_lens(
    x: jnp.ndarray, min_len, max_len, dtype: jnp.dtype = jnp.float32
):
    max_len = jnp.expand_dims(max_len, -1)
    min_len = jnp.expand_dims(min_len, -1)
    idx = jnp.broadcast_to(jnp.arange(x.shape[1]), (x.shape[0], x.shape[1]))
    return (jnp.less(idx, max_len) & jnp.greater_equal(idx, min_len)).astype(dtype)


def griff_lim_gpu(
    stft_matrix,
    n_iter=32,
    hop_length=256,
    win_length=1024,
    window="hanning",
    dtype=np.float32,
    pad_mode="reflect",
    momentum=0.99,
    init="random",
    random_state=None,
):
    """
    This mimic librosa version
    """
    if random_state is None:
        rng = np.random
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

    if momentum > 1:
        warnings.warn(
            "Griffin-Lim with momentum={} > 1 can be unstable. "
            "Proceed with caution!".format(momentum)
        )
    elif momentum < 0:
        raise ParameterError("griffinlim() called with momentum={} < 0".format(momentum))

    # Infer n_fft from the spectrogram shape
    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # using complex64 will keep the result to minimal necessary precision
    angles = np.empty(stft_matrix.shape, dtype=np.complex64)
    if init == "random":
        # randomly initialize the phase
        angles[:] = np.exp(2j * np.pi * rng.rand(*stft_matrix.shape))
    elif init is None:
        # Initialize an all ones complex matrix
        angles[:] = 1.0
    else:
        raise ParameterError("init={} must either None or 'random'".format(init))

    # And initialize the previous iterate to 0
    rebuilt = 0.0
    stft_jit = partial(jax.jit(stft, static_argnums=(1, 2, 3, 4, 5)))
    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt

        inverse = inverse_stft_gpu(
            stft_matrix * angles,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            dtype=dtype,
        )

        rebuilt = stft_jit(
            inverse,
            n_fft,
            hop_length,
            win_length,
            window,
            pad_mode,
        )

        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + 1e-16

    return inverse_stft_gpu(
        stft_matrix * angles,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        dtype=dtype,
    )


def power_to_db(power, ref=1.0, a_min=1e-10, top_db=80.0):
    if a_min <= 0:
        raise ParameterError("a_min must be strictly positive")

    if jnp.issubdtype(power.dtype, jnp.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead."
        )
        magnitude = jnp.abs(power)
    else:
        magnitude = power

    ref_value = jnp.abs(ref)

    log_spec = 10.0 * jnp.log10(jnp.maximum(a_min, magnitude))
    log_spec -= 10.0 * jnp.log10(jnp.maximum(a_min, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = jnp.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def amplitude_to_db(
    amplitude: jnp.ndarray, ref: float = 1.0, a_min: float = 1e-5, top_db: float = 80.0
) -> jnp.ndarray:
    """
    librosa based amplitude_to_db converted to jax

    Convert an amplitude spectrogram to dB-scaled spectrogram.

    This is equivalent to ``power_to_db(power**2)``, but is provided for convenience.

    Args:
        amplitude: input amplitude
        ref:
        a_min: minimum threshold for ``amplitude`` and ``ref``
        top_db:  float >= 0 [scalar]
        threshold the output at ``top_db`` below the peak:

    Returns:
        ``amplitude`` measured in dB

    """

    ref_value = jnp.abs(ref)

    magnitude = jnp.abs(amplitude)

    power = jnp.square(magnitude)

    return power_to_db(power, ref=ref_value ** 2, a_min=a_min ** 2, top_db=top_db)


def mag_phase(spectrogram: jnp.ndarray, power=1):
    """
    Separate a complex-valued spectrogram into its magnitude (S)
    and phase (P) components, so that ``spectrogram = S * P``.
    """

    mag = jnp.abs(spectrogram)
    mag **= power
    phase = jnp.exp(1.0j * jnp.angle(spectrogram))

    return mag, phase

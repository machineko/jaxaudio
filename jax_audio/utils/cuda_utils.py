from numba import cuda


@cuda.jit
def gpu_add_irfft(arr, tmp_ir, hop_len):
    """
    'fast' overlap add on gpu for output
    """
    x, y = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(x, tmp_ir.shape[0], stridex):

        for j in range(y, tmp_ir.shape[1], stridey):
            sample = j * hop_len
            index = sample + i
            cuda.atomic.add(arr, index, tmp_ir[i, j])


@cuda.jit
def gpu_add_win(arr, square_window, max_y, hop_len):
    """
    'fast' overlap add on gpu for fft_window
    """
    x, y = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(x, square_window.shape[0], stridex):

        for j in range(y, max_y, stridey):
            sample = j * hop_len
            index = sample + i
            cuda.atomic.add(arr, index, square_window[i])

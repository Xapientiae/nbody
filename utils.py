import numpy as np
import math
from numba import njit


def temperature_to_rgba(temp_k):
    """Mapowanie temperatura (K) -> kolor RGBA (zwraca (N,4))."""
    arr = np.atleast_1d(np.asarray(temp_k, dtype=float))
    return temperature_to_rgba_jit(arr)


@njit(fastmath=True)
def temperature_to_rgba_jit(t_arr):
    """Numba: oblicz RGBA dla tablicy temperatur (K)."""
    n = t_arr.size
    out = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        t = t_arr[i] / 100.0
        if t <= 66.0:
            r = 255.0
            tt = t if t > 1e-6 else 1e-6
            g = 99.4708025861 * math.log(tt) - 161.1195681661
            if t <= 19.0:
                b = 0.0
            else:
                b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
        else:
            r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
            g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
            b = 255.0
        # clamp 0..255
        if r < 0.0:
            r = 0.0
        elif r > 255.0:
            r = 255.0
        if g < 0.0:
            g = 0.0
        elif g > 255.0:
            g = 255.0
        if b < 0.0:
            b = 0.0
        elif b > 255.0:
            b = 255.0
        out[i, 0] = r / 255.0
        out[i, 1] = g / 255.0
        out[i, 2] = b / 255.0
        out[i, 3] = 1.0
    return out


def masses_to_rgba(masses):
    """Map masses (M_sun) -> RGBA array (N,4, float32)."""
    from physics import mass_to_temperature
    m_arr = np.asarray(masses, dtype=float)
    temps = mass_to_temperature(m_arr)
    rgba = temperature_to_rgba(temps)
    return np.asarray(rgba, dtype=np.float32)

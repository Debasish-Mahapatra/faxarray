"""Spectral transforms for FA fields.

Two coordinate systems are supported:

* Bi-Fourier (ALADIN/LAM): coefficients organised by meridional row, each
  coefficient a quartet ``(a, c, b, d)`` packing the symmetric Fourier modes.
  Forward and inverse transforms use a 2D FFT.
* Spherical harmonics (ARPEGE/global): triangular truncation T, complex
  coefficients ``a_{n,m} = a + i b`` ordered by zonal wavenumber m. Inverse
  transform combines an associated-Legendre evaluation per latitude with an
  inverse FFT along longitude. The forward transform is the dual operation
  with Gaussian quadrature weights.

The global implementation is a NumPy/SciPy reference. It is intentionally
straightforward; for bit-identical agreement with EPYGRAM/``ectrans4py``
the production path should still go through ``ectrans4py`` on Linux.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LamSpectralLayout:
    """ALADIN bi-Fourier coefficient layout: by meridian, quartets per zonal."""

    by_meridian: Tuple[Tuple[int, int, int], ...]
    coeff_count: int


def lam_sp2gp(coefficients: np.ndarray, layout: LamSpectralLayout, ny: int, nx: int) -> np.ndarray:
    """Inverse bi-Fourier transform: ALADIN spectral coefficients -> gridpoint."""

    scale = float(nx * ny)
    spectrum = np.zeros((ny, nx), dtype=np.complex128)

    for meridian, (start, _, max_zonal) in enumerate(layout.by_meridian):
        for zonal in range(max_zonal + 1):
            block = coefficients[start - 1 + 4 * zonal : start - 1 + 4 * zonal + 4]
            a = float(block[0])
            c = float(block[1])
            b = float(block[2])
            d = float(block[3])

            if zonal == 0 and meridian == 0:
                spectrum[0, 0] += scale * a
            elif meridian == 0:
                spectrum[0, zonal] += 0.5 * scale * (a - 1j * b)
                spectrum[0, (-zonal) % nx] += 0.5 * scale * (a + 1j * b)
            elif zonal == 0:
                spectrum[meridian, 0] += 0.5 * scale * (a - 1j * c)
                spectrum[(-meridian) % ny, 0] += 0.5 * scale * (a + 1j * c)
            else:
                spectrum[meridian, zonal] += 0.25 * scale * (a - d - 1j * (b + c))
                spectrum[(-meridian) % ny, zonal] += 0.25 * scale * (a + d - 1j * (b - c))
                spectrum[meridian, (-zonal) % nx] += 0.25 * scale * (a + d + 1j * (b - c))
                spectrum[(-meridian) % ny, (-zonal) % nx] += 0.25 * scale * (a - d + 1j * (b + c))

    return np.fft.ifft2(spectrum).real.astype(np.float64, copy=False)


def lam_gp2sp(values: np.ndarray, layout: LamSpectralLayout) -> np.ndarray:
    """Forward bi-Fourier transform: gridpoint values -> ALADIN spectral.

    Inverse of :func:`lam_sp2gp`. Assumes the input grid covers the C+I zone
    used by the spectral coefficient layout (i.e. comes back from
    ``lam_sp2gp``).
    """

    ny, nx = values.shape
    spectrum = np.fft.fft2(values)
    scale = float(nx * ny)
    coefficients = np.zeros(layout.coeff_count, dtype=np.float64)

    for meridian, (start, _, max_zonal) in enumerate(layout.by_meridian):
        for zonal in range(max_zonal + 1):
            base = start - 1 + 4 * zonal

            if zonal == 0 and meridian == 0:
                coefficients[base] = spectrum[0, 0].real / scale
            elif meridian == 0:
                pos = spectrum[0, zonal]
                neg = spectrum[0, (-zonal) % nx]
                coefficients[base] = (pos + neg).real / scale
                coefficients[base + 2] = (1j * (pos - neg)).real / scale
            elif zonal == 0:
                pos = spectrum[meridian, 0]
                neg = spectrum[(-meridian) % ny, 0]
                coefficients[base] = (pos + neg).real / scale
                coefficients[base + 1] = (1j * (pos - neg)).real / scale
            else:
                p = spectrum[meridian, zonal]
                q = spectrum[(-meridian) % ny, zonal]
                r = spectrum[meridian, (-zonal) % nx]
                s = spectrum[(-meridian) % ny, (-zonal) % nx]

                a = (p + q + r + s).real / scale
                d = (-p + q + r - s).real / scale
                b = (1j * (p + q - r - s)).real / scale
                c = (1j * (p - q + r - s)).real / scale

                coefficients[base] = a
                coefficients[base + 1] = c
                coefficients[base + 2] = b
                coefficients[base + 3] = d

    return coefficients


@dataclass(frozen=True)
class GaussSpectralLayout:
    """Triangular spectral coefficient layout for global ARPEGE fields.

    ``coeff_offsets`` maps zonal wavenumber ``m`` to the index of the first
    real coefficient in the flat coefficient array. For ``m == 0`` only
    real parts are stored (``T + 1`` reals); for ``m > 0`` the layout
    interleaves ``Re, Im`` for each ``n = m..T`` (``2 * (T - m + 1)`` reals).

    Total coefficient count is ``(T + 1) ** 2`` reals.
    """

    truncation: int
    coeff_offsets: Tuple[int, ...]
    total_real_coeffs: int


def make_gauss_layout(truncation: int) -> GaussSpectralLayout:
    """Build the standard FA "model" coefficient layout for triangular T."""

    offsets = []
    cursor = 0
    for m in range(truncation + 1):
        offsets.append(cursor)
        if m == 0:
            cursor += truncation + 1
        else:
            cursor += 2 * (truncation - m + 1)
    return GaussSpectralLayout(
        truncation=truncation,
        coeff_offsets=tuple(offsets),
        total_real_coeffs=cursor,
    )


def _associated_legendre(truncation: int, mu: float) -> np.ndarray:
    """Fully-normalised associated Legendre polynomials up to ``truncation``.

    Returns an array ``P[m, n]`` for ``0 <= m <= truncation`` and
    ``m <= n <= truncation`` (entries with ``n < m`` are zero). Uses the
    standard recurrence with the ``sqrt((2n+1)(n-m)!/(n+m)!)`` normalisation.
    """

    sin_t = math.sqrt(max(0.0, 1.0 - mu * mu))
    pmm = np.zeros((truncation + 1, truncation + 1), dtype=np.float64)

    pmm[0, 0] = 1.0 / math.sqrt(2.0)
    for m in range(1, truncation + 1):
        pmm[m, m] = pmm[m - 1, m - 1] * sin_t * math.sqrt((2.0 * m + 1.0) / (2.0 * m))

    for m in range(truncation):
        pmm[m, m + 1] = mu * math.sqrt(2.0 * m + 3.0) * pmm[m, m]
        for n in range(m + 2, truncation + 1):
            anm = math.sqrt((4.0 * n * n - 1.0) / (n * n - m * m))
            bnm = math.sqrt(((2.0 * n + 1.0) * ((n - 1.0) ** 2 - m * m)) / ((2.0 * n - 3.0) * (n * n - m * m)))
            pmm[m, n] = anm * mu * pmm[m, n - 1] - bnm * pmm[m, n - 2]

    return pmm


def gauss_sp2gp(
    coefficients: np.ndarray,
    layout: GaussSpectralLayout,
    sin_lats: np.ndarray,
    lon_number_by_lat: Sequence[int],
) -> np.ndarray:
    """Inverse spherical-harmonic transform: ARPEGE coefficients -> gridpoint.

    Returns a flat array with values laid out as ``[row_0, row_1, ...]``,
    one row per latitude, where row j has ``lon_number_by_lat[j]`` points.

    This is a NumPy reference implementation. For very large truncations or
    bit-identical match with ``ectrans``, prefer the Fortran path.
    """

    truncation = layout.truncation
    n_lat = len(sin_lats)
    total = int(sum(lon_number_by_lat))
    output = np.empty(total, dtype=np.float64)

    cursor = 0
    for j in range(n_lat):
        nlon = int(lon_number_by_lat[j])
        legendre = _associated_legendre(truncation, float(sin_lats[j]))

        fourier = np.zeros(nlon, dtype=np.complex128)
        for m in range(min(truncation, nlon // 2) + 1):
            offset = layout.coeff_offsets[m]
            poly = legendre[m, m:]
            if m == 0:
                real_part = coefficients[offset : offset + truncation + 1]
                fourier[0] = float(np.dot(poly, real_part))
            else:
                real_part = coefficients[offset : offset + 2 * (truncation - m + 1) : 2]
                imag_part = coefficients[offset + 1 : offset + 2 * (truncation - m + 1) : 2]
                re_sum = float(np.dot(poly, real_part))
                im_sum = float(np.dot(poly, imag_part))
                value = (re_sum + 1j * im_sum) * 0.5
                fourier[m] = value
                fourier[nlon - m] = np.conj(value)

        row = np.fft.ifft(fourier) * nlon
        output[cursor : cursor + nlon] = row.real
        cursor += nlon

    return output


def gauss_gp2sp(
    values: np.ndarray,
    layout: GaussSpectralLayout,
    sin_lats: np.ndarray,
    gauss_weights: np.ndarray,
    lon_number_by_lat: Sequence[int],
) -> np.ndarray:
    """Forward spherical-harmonic transform with Gaussian quadrature.

    ``gauss_weights`` are the standard Gaussian-Legendre weights for the
    given latitudes. Result is a flat real coefficient array matching
    :func:`gauss_sp2gp`.
    """

    truncation = layout.truncation
    n_lat = len(sin_lats)
    coefficients = np.zeros(layout.total_real_coeffs, dtype=np.float64)

    cursor = 0
    for j in range(n_lat):
        nlon = int(lon_number_by_lat[j])
        weight = float(gauss_weights[j])
        row = values[cursor : cursor + nlon]
        cursor += nlon

        fourier = np.fft.fft(row) / nlon
        legendre = _associated_legendre(truncation, float(sin_lats[j]))

        for m in range(min(truncation, nlon // 2) + 1):
            offset = layout.coeff_offsets[m]
            poly = legendre[m, m:]
            if m == 0:
                value = fourier[0]
                coefficients[offset : offset + truncation + 1] += poly * value.real * weight
            else:
                value = fourier[m] * 2.0
                coefficients[offset : offset + 2 * (truncation - m + 1) : 2] += (
                    poly * value.real * weight
                )
                coefficients[offset + 1 : offset + 2 * (truncation - m + 1) : 2] += (
                    poly * value.imag * weight
                )

    return coefficients


def gaussian_latitudes_and_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gaussian quadrature nodes (sin lat) and weights for ``n`` lats.

    Wraps :func:`numpy.polynomial.legendre.leggauss` so callers don't have
    to reach into NumPy directly. Result is ordered north-to-south.
    """

    nodes, weights = np.polynomial.legendre.leggauss(int(n))
    order = np.argsort(-nodes)
    return nodes[order].astype(np.float64), weights[order].astype(np.float64)

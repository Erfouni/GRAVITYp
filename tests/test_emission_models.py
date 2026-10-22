"""Regression tests for the piecewise emission profiles."""

import unittest

import numpy as np

from gravityp import (
    intensity_at_ISCO,
    intensity_at_inner_edge,
    intensity_at_photon_ring,
)


PROFILES = (
    (intensity_at_ISCO, 6.0, [5, 6, 7, 8], [0.0, 1.0, 0.25, 1 / 9]),
    (intensity_at_photon_ring, 3.0, [2, 3, 4, 5], [0.0, 1.0, 0.125, 1 / 27]),
    (
        intensity_at_inner_edge,
        5.0,
        [4, 5, 6, 7],
        [0.0, 1.0, 0.5, 0.2951672353008665],
    ),
)


class EmissionModelsTests(unittest.TestCase):
    def test_integer_arrays_keep_fractional_intensity(self):
        for profile, edge, radii, expected in PROFILES:
            for dtype in (np.int32, np.int64, np.uint64):
                with self.subTest(profile=profile.__name__, dtype=dtype):
                    actual = profile(np.array(radii, dtype=dtype), edge)
                    np.testing.assert_allclose(actual, expected)

    def test_floating_arrays_keep_expected_intensity(self):
        for profile, edge, radii, expected in PROFILES:
            for dtype in (np.float32, np.float64):
                with self.subTest(profile=profile.__name__, dtype=dtype):
                    actual = profile(np.array(radii, dtype=dtype), edge)
                    np.testing.assert_allclose(actual, expected)

    def test_integer_scalars_keep_fractional_intensity(self):
        for profile, edge, radii, expected in PROFILES:
            for scalar_type in (int, np.int64):
                with self.subTest(profile=profile.__name__, scalar_type=scalar_type):
                    actual = profile(scalar_type(radii[2]), edge)
                    self.assertEqual(actual.shape, ())
                    np.testing.assert_allclose(actual, expected[2])

    def test_sequences_keep_fractional_intensity(self):
        for profile, edge, radii, expected in PROFILES:
            for sequence_type in (list, tuple):
                with self.subTest(profile=profile.__name__, sequence_type=sequence_type):
                    actual = profile(sequence_type(radii), edge)
                    np.testing.assert_allclose(actual, expected)

    def test_multidimensional_arrays_keep_shape_and_intensity(self):
        for profile, edge, radii, expected in PROFILES:
            with self.subTest(profile=profile.__name__):
                actual = profile(np.array(radii).reshape(2, 2), edge)
                self.assertEqual(actual.shape, (2, 2))
                np.testing.assert_allclose(actual, np.array(expected).reshape(2, 2))

    def test_empty_arrays_return_empty_profiles(self):
        for profile, edge, _, _ in PROFILES:
            with self.subTest(profile=profile.__name__):
                actual = profile(np.array([], dtype=int), edge)
                self.assertEqual(actual.shape, (0,))

    def test_input_arrays_are_not_modified(self):
        for profile, edge, radii, _ in PROFILES:
            with self.subTest(profile=profile.__name__):
                original = np.array(radii, dtype=int)
                profile(original, edge)
                np.testing.assert_array_equal(original, radii)


if __name__ == "__main__":
    unittest.main()

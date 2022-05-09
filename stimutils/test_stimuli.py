""" Test stimuli module

Run tests with::

    python3 -m pytest stimutils/test_stimuli.py
"""

import os.path as op

import numpy as np
import numpy.testing as npt

from stimutils import stimuli


# Name of file to read for testing
COND_TEST_FNAME = 'cond_test1.txt'

# Get the file from the directory containing this test file.
MY_DIR = op.dirname(__file__)
COND_TEST_PATH = op.join(MY_DIR, COND_TEST_FNAME)


def test_events2neural():
    # test events2neural function
    # Read example file.
    neural = stimuli.events2neural(COND_TEST_PATH, 2, 16)
    # Expected values for tr=2, n_trs=16
    expected = np.zeros(16)
    expected[5:8] = 1
    expected[10:12] = 2
    expected[12] = 0.1
    npt.assert_array_equal(neural, expected)
    neural = stimuli.events2neural(COND_TEST_PATH, 1, 30)
    # Expected values for tr=1, n_trs=30
    expected = np.zeros(30)
    expected[10:16] = 1
    expected[20:24] = 2
    expected[24:26] = 0.1
    npt.assert_array_equal(neural, expected)

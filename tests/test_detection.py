import numpy as np

from gulfseis.detection import aic_pick, sta_lta, trigger_onsets


def test_sta_lta_detects_onset():
    rng = np.random.default_rng(0)
    fs = 100
    x = rng.standard_normal(6000)
    x[3000:3300] *= 20
    cft = sta_lta(x, fs, 1.0, 10.0)
    on = trigger_onsets(cft, 4.0, 1.5)
    assert len(on) == 1
    assert 3000 <= on[0][0] <= 3100


def test_aic_finds_onset():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(400)
    x[250:] *= 10
    assert abs(aic_pick(x) - 250) <= 3


def test_trigger_needs_off_before_retrigger():
    cft = np.array([0, 5, 5, 3, 5, 1, 0, 6, 0], float)
    assert trigger_onsets(cft, 4, 1.5) == [(1, 5), (7, 8)]

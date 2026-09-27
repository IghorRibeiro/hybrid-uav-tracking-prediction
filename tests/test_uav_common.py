"""Testes das metricas e do preditor Kalman com dados sinteticos.

Nao precisam do dataset nem de GPU. Rode com:
    python -m pytest tests/
ou, sem pytest:
    python tests/test_uav_common.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uav_common as u  # noqa: E402


def test_iou():
    assert u.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert u.iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    # metade sobreposta: intersecao 50, uniao 150
    assert math.isclose(u.iou((0, 0, 10, 10), (5, 0, 10, 10)), 50 / 150)


def test_tracking_metrics_with_absent_frames():
    gt = {1: (0, 0, 10, 10), 2: (0, 0, 10, 10), 3: None, 4: None}
    predictions = {
        1: (0, 0, 10, 10, 0.9, 5.0),    # acerto
        2: (8, 8, 10, 10, 0.9, 5.0),    # erro (IoU baixo)
        3: (0, 0, 10, 10, 0.1, 5.0),    # sem alvo, confianca baixa -> correto
        4: (0, 0, 10, 10, 0.9, 5.0),    # sem alvo, mas reportou alvo -> erro
    }
    metrics = u.tracking_metrics(gt, predictions)
    assert metrics["frames_total"] == 4
    assert math.isclose(metrics["SA"], 2 / 4)           # quadro 1 + quadro 3
    assert math.isclose(metrics["SR@0.5"], 1 / 2)       # so quadros com alvo
    assert math.isclose(metrics["FPS"], 200.0)


def test_missing_prediction_counts_as_zero_iou():
    gt = {1: (0, 0, 10, 10), 2: (0, 0, 10, 10)}
    predictions = {1: (0, 0, 10, 10, 0.9, 1.0)}
    metrics = u.tracking_metrics(gt, predictions)
    assert math.isclose(metrics["IoU_medio"], 0.5)
    assert math.isclose(metrics["SR@0.5"], 0.5)


def _linear_track(n=120, vx=3.0, vy=-1.5):
    return {f: (100.0 + vx * f, 400.0 + vy * f) for f in range(1, n + 1)}


def test_kalman_constant_velocity_is_accurate():
    centers = _linear_track()
    errors = u.prediction_errors(centers, centers, damping=1.0)
    # depois de convergir, velocidade constante e prevista quase sem erro
    late = errors[30][-20:]
    assert max(late) < 1.0


def test_damping_shortens_prediction():
    centers = _linear_track()
    persistent = u.prediction_errors(centers, centers, damping=1.0)
    damped = u.prediction_errors(centers, centers, damping=0.95)
    # num movimento retilineo o amortecimento so pode piorar a previsao longa
    assert np.mean(damped[30]) > np.mean(persistent[30])


def test_damping_applied_at_each_prediction_step():
    transition = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    state = np.array([[0.0], [0.0], [10.0], [0.0]], np.float32)
    predicted = u.forward(state, 3, transition, damping=0.5)
    # deslocamentos 10 + 5 + 2,5 = 17,5 (gamma aplicado a cada passo)
    assert math.isclose(float(predicted[0, 0]), 17.5, rel_tol=1e-6)
    assert math.isclose(float(u.forward(state, 3, transition, 1.0)[0, 0]), 30.0, rel_tol=1e-6)


def test_ci95_t_matches_paper_example():
    # H30 do hibrido no artigo: 139,9 +/- 97,4 px com n = 20 -> [94,3 ; 185,5]
    base = np.random.default_rng(0).normal(size=20)
    base = (base - base.mean()) / base.std(ddof=1)
    values = 139.9 + 97.4 * base
    low, high = u.ci95_t(values)
    assert round(low, 1) == 94.3 and round(high, 1) == 185.5


def test_official_gt_prefers_complete_folder(tmp_path=None):
    import tempfile
    root = Path(tmp_path or tempfile.mkdtemp())
    gt = root / "gt" / "Anti-UAV-Tracking-V0GT"
    gt.mkdir(parents=True)
    for i in (1, 2):
        (gt / f"video{i:02d}_gt.txt").write_text("1 2 3 4\n-100 -100 -100 -100\n")
    copy = root / "img" / "Anti-UAV-Tracking-V0" / "video01"
    copy.mkdir(parents=True)
    (copy / "video01_gt.txt").write_text("9 9 9 9\n")
    files = u.official_gt_files(root)
    assert set(files) == {"video01", "video02"}
    assert files["video01"].parent == gt
    frames = u.read_official_gt(files["video01"])
    assert frames == {1: (1.0, 2.0, 3.0, 4.0), 2: None}


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} testes passaram")

#!/usr/bin/env python3
"""Utilidades comuns do pipeline reproduzivel (Anti-VANT / DUT Anti-UAV).

Este modulo concentra TRES coisas que, se ficassem duplicadas em cada script,
acabariam divergindo entre si:

1. Leitura de dados: ground truth oficial, rotulos YOLO e arquivos de predicao.
2. Metricas de rastreamento: SA, SR@0.5, IoU medio, AUC.
3. Erro de predicao de trajetoria com Filtro de Kalman de velocidade constante
   (CV), com ou sem amortecimento de velocidade (damping).

Convencoes fixadas aqui e usadas por TODOS os scripts deste pipeline:

- Coordenadas em pixels, no espaco da imagem original de cada sequencia
  (1920x1080 ou 1280x720, conforme o video: ver video_resolution()).
- Caixa no formato (x, y, w, h), com (x, y) no canto superior esquerdo.
- O ground truth oficial do DUT Anti-UAV usa (x, y, w, h) por linha e marca
  quadro sem alvo como "-100 -100 -100 -100".
- Horizonte H significa "H quadros a frente": H30 em t compara a predicao com
  o ground truth de t+30.
- O rastreador e avaliado como rastreador de ALVO UNICO: em cada quadro usamos
  a caixa de maior confianca, como fazem os scripts historicos do projeto.
- Quadro sem alvo conta para a SA: acertar significa nao reportar alvo (ou
  reportar com confianca baixa).
"""

from __future__ import annotations

import csv
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

# Resolucao de ultimo recurso, usada apenas se nem o resolutions.json nem os
# quadros estiverem disponiveis. ATENCAO: a resolucao VARIA entre as sequencias
# do DUT Anti-UAV (a maioria e 1920x1080, mas por exemplo video03 e 1280x720),
# por isso ela nunca deve ser assumida: ver video_resolution().
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Horizontes de predicao avaliados (em quadros).
HORIZONS = (5, 10, 15, 30)

# Limiares de IoU da curva de sucesso (21 pontos, passo 0.05 -- convencao do
# toolkit OTB, a mesma usada nos scripts historicos do projeto).
IOU_THRESHOLDS = np.linspace(0.0, 1.0, 21)

# Confianca abaixo da qual uma predicao e tratada como "sem alvo".
CONF_ABSENT = 0.25

TARGET_ABSENT = (-100.0, -100.0, -100.0, -100.0)


# --------------------------------------------------------------------------
# Descoberta de arquivos
# --------------------------------------------------------------------------

def list_videos(images_dir: Path) -> list[str]:
    """IDs das sequencias presentes num diretorio de quadros (videoNN_00001.jpg)."""
    found = {p.name.split("_")[0] for p in images_dir.glob("video*_*.jpg")}
    return sorted(found)


def frames_of_video(images_dir: Path, video_id: str) -> list[Path]:
    """Quadros de uma sequencia, em ordem numerica (ignora os '._*' do macOS)."""
    frames = [p for p in images_dir.glob(f"{video_id}_*.jpg") if not p.name.startswith("._")]
    return sorted(frames, key=frame_number)


def frame_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


# --------------------------------------------------------------------------
# Resolucao por sequencia
# --------------------------------------------------------------------------

_WARNED_RESOLUTION = False


def jpeg_size(path: Path) -> tuple[int, int] | None:
    """Le largura e altura de um JPEG sem carregar a imagem."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    index = 2
    while index < len(data) - 9:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB):
            height, width = struct.unpack(">HH", data[index + 5:index + 9])
            return int(width), int(height)
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        index += 2 + struct.unpack(">H", data[index + 2:index + 4])[0]
    return None


def measure_video_size(gt_file: Path) -> tuple[int, int] | None:
    """Resolucao real de uma sequencia, medida do primeiro quadro que existir.

    As pastas de imagens recebem varios nomes conforme a origem (img/videoNN,
    videoNN, quadros planos videoNN_00001.jpg), entao procuramos por todos.
    """
    video_id = gt_file.stem.replace("_gt", "")
    base = gt_file.parent
    patterns = (
        f"video*/{video_id}/{video_id}_*.jpg",
        f"img/{video_id}/*.jpg",
        f"images/{video_id}/*.jpg",
        f"{video_id}/*.jpg",
        f"{video_id}_*.jpg",
        f"frames/{video_id}/*.jpg",
    )
    for root in (base, base.parent):
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                size = jpeg_size(matches[0])
                if size:
                    return size
        for candidate in sorted(root.rglob(f"{video_id}_*.jpg"))[:1]:
            size = jpeg_size(candidate)
            if size:
                return size
        for candidate in sorted(p for p in root.rglob(video_id) if p.is_dir())[:1]:
            frames = sorted(candidate.glob("*.jpg"))[:1]
            if frames:
                size = jpeg_size(frames[0])
                if size:
                    return size
    return None


def write_resolutions(path: Path, sizes: dict[str, tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {video: {"width": int(w), "height": int(h)} for video, (w, h) in sorted(sizes.items())}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def video_resolution(
    video_id: str,
    size: tuple[int, int] | None = None,
    images_dir: Path | None = None,
    resolutions_file: Path | None = None,
) -> tuple[int, int]:
    """Resolucao de uma sequencia, na ordem de confiabilidade:

    1. o tamanho medido agora no proprio quadro (o dado real);
    2. o resolutions.json gravado no PASSO 2;
    3. 1920x1080 apenas como ultimo recurso, com aviso impresso uma vez.

    Isso importa porque as sequencias NAO tem todas a mesma resolucao: usar
    1920x1080 num video 1280x720 desloca todos os rotulos (~50% de erro de
    posicao) e contamina qualquer metrica calculada a partir deles.
    """
    if size is not None:
        return size
    for candidate in (resolutions_file,
                      Path("datasets/anti_uav_tracking_test_final/resolutions.json")):
        if candidate is not None and candidate.exists():
            try:
                entry = json.loads(candidate.read_text(encoding="utf-8")).get(video_id)
            except (json.JSONDecodeError, OSError):
                entry = None
            if entry:
                return int(entry["width"]), int(entry["height"])
    if images_dir is not None:
        frames = frames_of_video(images_dir, video_id)
        if frames:
            measured = jpeg_size(frames[0])
            if measured:
                return measured
    global _WARNED_RESOLUTION
    if not _WARNED_RESOLUTION:
        print(f"AVISO: resolucao de {video_id} desconhecida; assumindo 1920x1080. "
              "Rode o PASSO 2 para gravar datasets/anti_uav_tracking_test_final/resolutions.json.")
        _WARNED_RESOLUTION = True
    return IMG_WIDTH, IMG_HEIGHT


# --------------------------------------------------------------------------
# Leitura de anotacoes
# --------------------------------------------------------------------------

def official_gt_files(root: Path) -> dict[str, Path]:
    """Mapeia videoNN -> arquivo videoNN_gt.txt oficial, procurando em toda a arvore.

    O zip oficial do ground truth (Anti-UAV-Tracking-V0GT.zip) extrai para uma
    SUBPASTA, e o pacote de imagens tambem traz um videoNN_gt.txt dentro de cada
    pasta videoNN/. Por isso a busca e recursiva e, havendo copias, da
    preferencia a pasta com mais arquivos de ground truth (a do pacote GT).
    """
    root = Path(root)
    if root.is_file():
        root = root.parent
    directories = sorted({p.parent for p in root.rglob("video*_gt.txt")},
                         key=lambda d: (-len(list(d.glob("video*_gt.txt"))), str(d)))
    files: dict[str, Path] = {}
    for directory in directories:
        for gt_file in sorted(directory.glob("video*_gt.txt")):
            if gt_file.stem.endswith("_first") or gt_file.name.startswith("._"):
                continue
            files.setdefault(gt_file.stem.replace("_gt", ""), gt_file)
    return dict(sorted(files.items()))


def require_official_gt(root: Path) -> dict[str, Path]:
    """Como official_gt_files, mas interrompe com mensagem util se nao achar nada.

    Falhar aqui e proposital: seguir adiante com ground truth vazio produziria
    metricas zeradas com cara de resultado.
    """
    files = official_gt_files(root)
    if not files:
        raise SystemExit(
            f"Nao encontrei nenhum video*_gt.txt dentro de {Path(root).resolve()}.\n"
            "Rode o PASSO 1 (src/01_baixar_dataset.py) ou aponte --gt-dir para a\n"
            "pasta que contem os arquivos de ground truth."
        )
    return files


def read_official_gt(gt_file: Path) -> dict[int, tuple[float, float, float, float] | None]:
    """Le videoNN_gt.txt oficial: x y w h por linha; -100 marca quadro sem alvo."""
    frames: dict[int, tuple[float, float, float, float] | None] = {}
    if not gt_file.exists():
        return frames
    for index, raw in enumerate(gt_file.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.split()
        if len(parts) != 4:
            continue
        x, y, w, h = (float(v) for v in parts)
        if x < 0 or y < 0 or w < 0 or h < 0:
            frames[index] = None
        else:
            frames[index] = (x, y, w, h)
    return frames


def read_yolo_labels(
    labels_dir: Path,
    video_id: str,
    size: tuple[int, int] | None = None,
    resolutions_file: Path | None = None,
) -> dict[int, tuple[float, float, float, float] | None]:
    """Le os rotulos YOLO (videoNN_id.txt) e converte de volta para pixels.

    Arquivo vazio significa "quadro sem alvo" (convencao do dataset convertido).
    A resolucao vem por sequencia (resolutions.json do PASSO 2), nunca de um
    valor fixo, porque as sequencias do DUT Anti-UAV diferem entre si.
    """
    frames: dict[int, tuple[float, float, float, float] | None] = {}
    if resolutions_file is None:
        resolutions_file = labels_dir.parent.parent / "resolutions.json"
    width, height = video_resolution(video_id, size=size, resolutions_file=resolutions_file)
    for label_path in sorted(labels_dir.glob(f"{video_id}_*.txt")):
        frame = int(label_path.stem.rsplit("_", 1)[1])
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            frames[frame] = None
            continue
        values = lines[0].split()
        if len(values) < 5:
            frames[frame] = None
            continue
        cx, cy, w, h = (float(v) for v in values[1:5])
        frames[frame] = (
            (cx - w / 2.0) * width,
            (cy - h / 2.0) * height,
            w * width,
            h * height,
        )
    return frames


def read_predictions(path: Path) -> dict[int, tuple[float, float, float, float, float, float]]:
    """Le um arquivo de predicao: frame,x,y,w,h,conf[,time_ms] (uma linha por quadro)."""
    predictions: dict[int, tuple[float, float, float, float, float, float]] = {}
    if not path.exists():
        return predictions
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
        if len(parts) < 6:
            parts = raw.split()
        if len(parts) < 6:
            continue
        try:
            frame = int(float(parts[0]))
            x, y, w, h, conf = (float(v) for v in parts[1:6])
            time_ms = float(parts[6]) if len(parts) >= 7 else math.nan
        except ValueError:
            continue
        predictions.setdefault(frame, (x, y, w, h, conf, time_ms))
    return predictions


def write_predictions(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Metricas de rastreamento
# --------------------------------------------------------------------------

def iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def tracking_metrics(gt_frames, predictions) -> dict:
    """Calcula SA, SR@0.5, IoU medio e AUC de uma sequencia.

    SA (State Accuracy, convencao do artigo):
        (quadros com alvo e IoU >= 0.5) + (quadros sem alvo corretamente
        declarados sem alvo) / total de quadros da sequencia.
    SR@0.5:
        fracao dos quadros COM alvo em que a predicao atinge IoU >= 0.5.
    IoU medio:
        media do IoU nos quadros com alvo (sem predicao conta como IoU 0).
    AUC:
        area sob a curva de sucesso (SR em funcao do limiar de IoU, 0 a 1).
    """
    frames = sorted(set(gt_frames) | set(predictions))
    total = len(frames)
    present = [f for f in frames if gt_frames.get(f) is not None]
    absent = [f for f in frames if gt_frames.get(f) is None]

    ious = []
    for frame in present:
        prediction = predictions.get(frame)
        ious.append(iou(gt_frames[frame], prediction[:4]) if prediction is not None else 0.0)

    sa_target = sum(1 for value in ious if value >= 0.5)
    sa_absent = 0
    for frame in absent:
        prediction = predictions.get(frame)
        if prediction is None or prediction[4] < CONF_ABSENT:
            sa_absent += 1

    sr_curve = {th: (sum(1 for v in ious if v >= th) / len(ious) if ious else 0.0) for th in IOU_THRESHOLDS}
    auc = float(np.trapezoid(np.array(list(sr_curve.values())), IOU_THRESHOLDS)) if ious else 0.0

    times = [p[5] for p in predictions.values() if not math.isnan(p[5])]
    coverage = len(predictions) / total if total else 0.0

    return {
        "frames_total": total,
        "frames_com_alvo": len(present),
        "frames_sem_alvo": len(absent),
        "predicoes": len(predictions),
        "cobertura": coverage,
        "SA": (sa_target + sa_absent) / total if total else 0.0,
        "SR@0.5": (sa_target / len(ious)) if ious else 0.0,
        "IoU_medio": float(np.mean(ious)) if ious else 0.0,
        "AUC": auc,
        "tempo_medio_ms": float(np.mean(times)) if times else math.nan,
        "FPS": 1000.0 / float(np.mean(times)) if times and np.mean(times) > 0 else math.nan,
    }


# --------------------------------------------------------------------------
# Predicao de trajetoria (Kalman CV, com amortecimento opcional)
# --------------------------------------------------------------------------

def init_kalman(cx: float, cy: float) -> cv2.KalmanFilter:
    """Filtro de Kalman de velocidade constante: estado [x, y, vx, vy]."""
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    state = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
    kalman.statePre = state.copy()
    kalman.statePost = state.copy()
    return kalman


def forward(state: np.ndarray, horizon: int, transition: np.ndarray, damping: float = 1.0) -> np.ndarray:
    """Propaga o estado horizon quadros a frente.

    Fiel ao projeto original (scripts/avaliacao_avancada/validacao_cv_damped.py):
    o amortecimento e aplicado A CADA passo da previsao, nao apenas no estado
    corrigido -- e essa aplicacao repetida que caracteriza o Kalman Damped.
    """
    predicted = state.copy()
    for _ in range(horizon):
        predicted = transition @ predicted
        if damping != 1.0:
            predicted[2, 0] *= damping
            predicted[3, 0] *= damping
    return predicted


def prediction_errors(centers, gt_centers, damping: float = 1.0, horizons=HORIZONS) -> dict:
    """Erro de predicao, em pixels, para cada horizonte.

    centers     : {frame: (cx, cy)} obtidos do rastreador.
    gt_centers  : {frame: (cx, cy)} do ground truth (apenas quadros com alvo).
    damping     : fator multiplicado por (vx, vy) apos cada correcao. 1.0 =
                  Kalman persistente (sem amortecimento); 0.95 = Kalman Damped
                  reportado no artigo.
    """
    errors = {h: [] for h in horizons}
    kalman = None
    transition = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    for frame in sorted(centers):
        cx, cy = centers[frame]
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        if kalman is None:
            kalman = init_kalman(cx, cy)
            continue

        kalman.predict()
        kalman.correct(measurement)
        if damping != 1.0:
            damped = kalman.statePost.copy()
            damped[2, 0] *= damping
            damped[3, 0] *= damping
            kalman.statePost = damped

        state = kalman.statePost.copy()
        for horizon in horizons:
            target = gt_centers.get(frame + horizon)
            if target is None:
                continue
            predicted = forward(state, horizon, transition, damping)
            errors[horizon].append(float(np.hypot(predicted[0, 0] - target[0], predicted[1, 0] - target[1])))

    return errors


def ci95_t(values) -> tuple[float, float]:
    """IC95% da media entre sequencias (t de Student), como reportado no artigo.

    Ex.: H30 do hibrido = 139,9 +/- 97,4 px (n = 20) -> [94,3 ; 185,5] px.
    """
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if array.size < 2:
        return math.nan, math.nan
    try:
        from scipy import stats
        multiplier = float(stats.t.ppf(0.975, array.size - 1))
    except ImportError:  # pragma: no cover - scipy e dependencia do pipeline
        multiplier = 1.96
    half = multiplier * float(array.std(ddof=1)) / math.sqrt(array.size)
    return float(array.mean() - half), float(array.mean() + half)


def summarize_errors(errors) -> dict:
    """Media, desvio, mediana, p95 e maximo por horizonte."""
    out = {}
    for horizon, values in errors.items():
        if values:
            array = np.asarray(values, dtype=float)
            out[f"H{horizon}_media"] = float(array.mean())
            out[f"H{horizon}_std"] = float(array.std(ddof=1)) if array.size > 1 else 0.0
            out[f"H{horizon}_mediana"] = float(np.median(array))
            out[f"H{horizon}_p95"] = float(np.percentile(array, 95))
            out[f"H{horizon}_max"] = float(array.max())
            out[f"H{horizon}_n"] = int(array.size)
        else:
            for suffix in ("media", "std", "mediana", "p95", "max"):
                out[f"H{horizon}_{suffix}"] = math.nan
            out[f"H{horizon}_n"] = 0
    return out


def centers_from_gt(gt_frames) -> dict:
    return {
        frame: (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        for frame, box in gt_frames.items()
        if box is not None
    }


def centers_from_predictions(predictions) -> dict:
    return {
        frame: (prediction[0] + prediction[2] / 2.0, prediction[1] + prediction[3] / 2.0)
        for frame, prediction in predictions.items()
    }

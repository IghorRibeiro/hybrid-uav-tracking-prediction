#!/usr/bin/env python3
"""PASSO 4 - Rodar os rastreadores e gravar as predicoes.

Todo rastreador escreve o MESMO formato, um arquivo por sequencia:

    results/predictions/<metodo>/videoNN.txt
    frame,x,y,w,h,confianca,tempo_ms

Isso e o que garante a comparacao justa: como todos leem os mesmos JPEGs, as
20 sequencias e os 24.804 quadros sao exatamente os mesmos para todos os
metodos -- sem depender de recortes de video diferentes.

Metodos disponiveis:

  hibrido    YOLO26n + rastreador local (CSRT por padrao), com redeteccao a
             cada --period quadros (o metodo proposto;
             --local-tracker csrt|kcf|mosse|medianflow). O filtro de Kalman
             consome apenas o centro da caixa de cada quadro, entao ele e
             aplicado no PASSO 5 sobre a trajetoria gravada aqui -- o resultado
             e o mesmo de roda-lo dentro do laco, e assim TODOS os metodos
             passam exatamente pelo mesmo preditor.
  botsort    Ultralytics BoT-SORT
  bytetrack  Ultralytics ByteTrack
  deepsort   deep-sort-realtime (requer: pip install deep-sort-realtime)
  ostrack / odtrack
             NAO executados aqui: vem de repositorios externos com pesos e
             ambiente proprios. Gere as predicoes com
             src/extras/rodar_ostrack_odtrack.py (mesmo formato) ou importe
             arquivos ja prontos com --importar (use --posicoes se estiverem no
             formato frame,x,y,w,h).

Protocolo de alvo unico: a sequencia tem UM drone; em cada quadro usamos a
caixa de maior confianca. Quadro em que o rastreador nao reporta nada e tratado
como "sem alvo" pelo PASSO 5.

Exemplos:
    # o metodo proposto, nas 20 sequencias
    python src/04_rastreadores.py --tracker hibrido --local-tracker csrt --device mps

    # os baselines de deteccao+associacao
    python src/04_rastreadores.py --tracker botsort
    python src/04_rastreadores.py --tracker bytetrack
    python src/04_rastreadores.py --tracker deepsort --device cpu

    # um unico video, para testar rapido
    python src/04_rastreadores.py --tracker hibrido --video video01

    # incorporar predicoes de OSTrack/ODTrack geradas em outro ambiente
    python src/04_rastreadores.py --tracker ostrack --importar /caminho/da/pasta/ostrack
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from uav_common import frame_number, frames_of_video, list_videos, write_predictions

TRACKER_CHOICES = ("hibrido", "botsort", "bytetrack", "deepsort", "ostrack", "odtrack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracker", choices=TRACKER_CHOICES, required=True)
    parser.add_argument("--images-dir", type=Path, default=Path("datasets/anti_uav_tracking_test_final/images/test"))
    parser.add_argument("--model", type=Path, default=Path("pesos_treinados/yolo26n_dut_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Padrao: results/predictions/<metodo>")
    parser.add_argument("--device", default="auto",
                        help="auto (padrao), mps, cpu, 0 ... -- auto escolhe CUDA > MPS > CPU. "
                             "O artigo mediu no Apple M2 (mps).")
    parser.add_argument("--video", default=None, help="Processar apenas uma sequencia (ex.: video01).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confianca das deteccoes dos baselines.")
    parser.add_argument("--init-conf", type=float, default=0.50, help="Confianca minima para inicializar o hibrido.")
    parser.add_argument("--switch-conf", type=float, default=0.70, help="Confianca minima para reancorar o hibrido.")
    parser.add_argument("--period", type=int, default=15, help="N_de_red: quadros entre redeteccoes do hibrido.")
    parser.add_argument("--local-tracker", default="csrt", choices=("csrt", "kcf", "mosse", "medianflow"))
    parser.add_argument("--importar", type=Path, default=None,
                        help="Pasta com .txt de predicoes ja calculadas (um por video).")
    parser.add_argument("--posicoes", action="store_true",
                        help="Com --importar: os arquivos estao em frame,x,y,w,h (sem confianca/tempo).")
    return parser.parse_args()


# ------------------------------------------------------------------ hibrido

def make_local_tracker(name: str):
    factories = {
        "csrt": ("TrackerCSRT_create", "TrackerCSRT_create"),
        "kcf": ("TrackerKCF_create", "TrackerKCF_create"),
        "mosse": ("TrackerMOSSE_create", "TrackerMOSSE_create"),
        "medianflow": ("TrackerMedianFlow_create", "TrackerMedianFlow_create"),
    }
    modern, legacy = factories[name]
    if hasattr(cv2, modern):
        return getattr(cv2, modern)()
    legacy_api = getattr(cv2, "legacy", None)
    if legacy_api is not None and hasattr(legacy_api, legacy):
        return getattr(legacy_api, legacy)()
    raise RuntimeError(
        f"{name.upper()} nao esta disponivel nesta instalacao do OpenCV "
        f"({cv2.__version__}). No opencv-python-headless os rastreadores sao "
        "limitados; instale 'opencv-contrib-python'."
    )


def detect_best(detector, frame, device: str, confidence: float):
    """Detecta o drone: devolve a caixa de maior confianca ou None."""
    results = detector(frame, device=device, conf=confidence, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    confidences = boxes.conf.detach().cpu().numpy()
    best = int(np.argmax(confidences))
    x1, y1, x2, y2 = boxes.xyxy[best].detach().cpu().numpy().tolist()
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    w, h = max(1, int(round(x2)) - x1), max(1, int(round(y2)) - y1)
    return (x1, y1, w, h, float(confidences[best]))


def run_hibrido(args, image_paths, output_path: Path) -> dict:
    from ultralytics import YOLO

    detector = YOLO(str(args.model.resolve()))
    rows = []
    times = []
    detector_calls = 0
    tracker = None
    last_conf = 0.0

    for index, path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        start = time.perf_counter()
        current = None

        if tracker is None:
            detector_calls += 1
            current = detect_best(detector, frame, args.device, args.init_conf)
            if current is not None:
                x, y, w, h, last_conf = current
                tracker = make_local_tracker(args.local_tracker)
                tracker.init(frame, (x, y, w, h))
        else:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = (int(round(v)) for v in box)
                w, h = max(1, w), max(1, h)
                current = (x, y, w, h, last_conf)
                if index % args.period == 0:
                    detector_calls += 1
                    candidate = detect_best(detector, frame, args.device, args.switch_conf)
                    if candidate is not None and candidate[4] >= args.switch_conf:
                        x, y, w, h, last_conf = candidate
                        tracker = make_local_tracker(args.local_tracker)
                        tracker.init(frame, (x, y, w, h))
                        current = candidate
            else:
                tracker = None

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)
        if current is not None:
            x, y, w, h, conf = current
            rows.append((frame_number(path), x, y, w, h, f"{conf:.6f}", f"{elapsed_ms:.3f}"))

    write_predictions(output_path, rows)
    mean_ms = float(np.mean(times)) if times else float("nan")
    return {
        "frames": len(image_paths), "predictions": len(rows), "detector_calls": detector_calls,
        "mean_time_ms": mean_ms, "fps": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
    }


# ---------------------------------------------------------------- baselines

def _xyxy_to_xywh(box) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def run_ultralytics_tracker(args, image_paths, output_path: Path, tracker_yaml: str) -> dict:
    """BoT-SORT / ByteTrack via Ultralytics, com estado persistente por sequencia."""
    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    rows = []
    times = []
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        start = time.perf_counter()
        result = model.track(frame, persist=True, tracker=tracker_yaml, conf=args.conf,
                             device=args.device, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            continue
        confidences = boxes.conf.detach().cpu().numpy()
        best = int(np.argmax(confidences))
        x, y, w, h = _xyxy_to_xywh(boxes.xyxy[best].detach().cpu().numpy())
        rows.append((frame_number(path), f"{x:.3f}", f"{y:.3f}", f"{w:.3f}", f"{h:.3f}",
                     f"{float(confidences[best]):.6f}", f"{elapsed_ms:.3f}"))

    write_predictions(output_path, rows)
    mean_ms = float(np.mean(times)) if times else float("nan")
    return {"frames": len(image_paths), "predictions": len(rows), "detector_calls": len(image_paths),
            "mean_time_ms": mean_ms, "fps": 1000.0 / mean_ms if mean_ms > 0 else float("nan")}


def run_deepsort(args, image_paths, output_path: Path) -> dict:
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError as error:
        raise SystemExit(
            "DeepSORT requer o pacote deep-sort-realtime:\n"
            "    pip install deep-sort-realtime\n"
            f"(import falhou: {error})"
        ) from error
    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.7,
                       nn_budget=None, embedder="mobilenet", embedder_gpu=False, half=False)
    rows = []
    times = []
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        start = time.perf_counter()
        result = model(frame, conf=args.conf, device=args.device, verbose=False)[0]
        detections = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            for coords, score, cls in zip(xyxy, confidences, classes):
                detections.append((list(_xyxy_to_xywh(coords)), float(score), int(cls)))
        tracks = tracker.update_tracks(detections, frame=frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)

        active = [t for t in tracks if t.is_confirmed() and t.time_since_update <= 1]
        if not active:
            continue
        active.sort(key=lambda t: (t.get_det_conf() or 0.0, -t.time_since_update), reverse=True)
        chosen = active[0]
        x, y, w, h = _xyxy_to_xywh(chosen.to_ltrb())
        rows.append((frame_number(path), f"{x:.3f}", f"{y:.3f}", f"{w:.3f}", f"{h:.3f}",
                     f"{float(chosen.get_det_conf() or 0.0):.6f}", f"{elapsed_ms:.3f}"))

    write_predictions(output_path, rows)
    mean_ms = float(np.mean(times)) if times else float("nan")
    return {"frames": len(image_paths), "predictions": len(rows), "detector_calls": len(image_paths),
            "mean_time_ms": mean_ms, "fps": 1000.0 / mean_ms if mean_ms > 0 else float("nan")}


def import_external(source: Path, output_dir: Path, only: str | None, positions_only: bool) -> int:
    """Copia/adapta predicoes geradas por repositorios externos (OSTrack, ODTrack)."""
    files = sorted(source.glob("*.txt")) if source.is_dir() else [source]
    if not files:
        raise SystemExit(f"Nenhum .txt encontrado em {source}")
    count = 0
    for path in files:
        video_id = path.stem
        if only and video_id != only:
            continue
        output_path = output_dir / f"{video_id}.txt"
        if not positions_only:
            shutil.copy2(path, output_path)
            count += 1
            continue
        # Formato externo frame,x,y,w,h -> acrescenta confianca 1.0 e tempo vazio.
        rows = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            parts = [p for p in raw.replace(";", ",").replace("\t", ",").split(",") if p.strip()]
            if len(parts) < 5:
                continue
            rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip(),
                         parts[3].strip(), parts[4].strip(), "1.000000", ""))
        write_predictions(output_path, rows)
        count += 1
    return count


def resolve_device(requested: str) -> str:
    """'auto' -> CUDA (0) se houver, senao MPS (Apple Silicon), senao CPU."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    output_dir = (args.output_dir or Path("results/predictions") / args.tracker).resolve()
    if (args.tracker == "hibrido" and args.output_dir is None
            and args.local_tracker != "csrt"):
        # Sem isto, os quatro rastreadores locais escreveriam na MESMA pasta
        # (results/predictions/hibrido/) e o PASSO 5 compararia CSRT com CSRT.
        output_dir = Path("results/predictions") / f"hibrido_{args.local_tracker}"
        output_dir = output_dir.resolve()
        print(f"Saida separada por rastreador local: {output_dir}")

    if args.tracker in ("ostrack", "odtrack"):
        if args.importar is None:
            raise SystemExit(
                f"O rastreador '{args.tracker}' vem de um repositorio externo (OSTrack/ODTrack),\n"
                "com pesos e ambiente proprios; ele nao e executado por este script.\n\n"
                "Opcao 0: rode src/extras/rodar_ostrack_odtrack.py (grava direto em results/predictions).\n"
                "Opcao 1: gere as predicoes naquele repositorio e importe assim:\n"
                f"    python src/04_rastreadores.py --tracker {args.tracker} "
                f"--importar /caminho/das/predicoes\n\n"
                "Opcao 2: use '--posicoes' se os arquivos tiverem apenas frame,x,y,w,h.\n\n"
                "Se voce nao rodou OSTrack/ODTrack, compare apenas os metodos que rodou:\n"
                "o PASSO 5 aceita qualquer subconjunto de metodos."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        count = import_external(args.importar.resolve(), output_dir, args.video, args.posicoes)
        print(f"{count} arquivos importados em {output_dir}")
        return

    if not images_dir.is_dir():
        raise SystemExit(
            f"Imagens nao encontradas em {images_dir}.\n"
            "Rode o PASSO 2 primeiro: python src/02_converter_dataset.py --only tracking"
        )

    videos = [args.video] if args.video else list_videos(images_dir)
    if not videos:
        raise SystemExit(f"Nenhuma sequencia video*_*.jpg em {images_dir}")

    if args.tracker in ("botsort", "bytetrack", "deepsort"):
        model_path = args.model.resolve()
        if not model_path.exists():
            raise SystemExit(
                f"Pesos nao encontrados: {model_path}\n"
                "Use os pesos do artigo (pesos_treinados/yolo26n_dut_best.pt) ou passe --model "
                "com o caminho do best.pt que voce treinou no PASSO 3."
            )
        print(f"Metodo: {args.tracker}   detector: {model_path.name}   "
              f"dispositivo: {resolve_device(args.device)}")

    args.device = resolve_device(args.device)
    if args.tracker == "hibrido":
        model_path = args.model.resolve()
        if not model_path.exists():
            raise SystemExit(
                f"Pesos nao encontrados: {model_path}\n"
                "Use os pesos do artigo (pesos_treinados/yolo26n_dut_best.pt, versionados no "
                "repositorio) ou treine os seus no PASSO 3 e passe --model."
            )
        print(f"Metodo: hibrido ({args.local_tracker.upper()}, N_red={args.period})   "
              f"detector: {model_path.name}   dispositivo: {args.device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    print(f"Metodo: {args.tracker}   sequencias: {len(videos)}   saida: {output_dir}")

    for video_id in videos:
        paths = frames_of_video(images_dir, video_id)
        if not paths:
            print(f"  AVISO sem quadros para {video_id}")
            continue
        if args.tracker == "hibrido":
            stats = run_hibrido(args, paths, output_dir / f"{video_id}.txt")
        elif args.tracker == "botsort":
            stats = run_ultralytics_tracker(args, paths, output_dir / f"{video_id}.txt", "botsort.yaml")
        elif args.tracker == "bytetrack":
            stats = run_ultralytics_tracker(args, paths, output_dir / f"{video_id}.txt", "bytetrack.yaml")
        else:
            stats = run_deepsort(args, paths, output_dir / f"{video_id}.txt")
        stats["video"] = video_id
        summaries.append(stats)
        print(f"  {video_id}: {stats['frames']} quadros, {stats['predictions']} predicoes, "
              f"{stats['fps']:.1f} FPS")

    summary_path = output_dir / "run_summary.csv"
    if summaries:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\nResumo do tempo de execucao: {summary_path}")
    print("Proximo passo: PASSO 5 (python src/05_avaliar.py --metodos ...)")


if __name__ == "__main__":
    main()

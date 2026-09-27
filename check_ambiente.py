#!/usr/bin/env python3
"""Checagem de ambiente antes de rodar o pipeline (nao treina nada, so verifica).

Responde a pergunta "por que nao deu igual?": lista versao de Python, PyTorch,
OpenCV, Ultralytics, dispositivo disponivel (CUDA/MPS/CPU), rastreadores locais
do OpenCV, resolucao lida do dataset e contagem de sequencias/quadros.

Exemplos:
    python check_ambiente.py
    python check_ambiente.py --recomendado     # mostra as versoes usadas no artigo
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

# Ambiente do autor (versoes instaladas no ambiente em que o artigo foi produzido).
RECOMMENDED = {
    "python": "3.13",
    "torch": "2.12.0",
    "torchvision": "0.27.0",
    "ultralytics": "8.4.53",
    "opencv-contrib": "4.13.0.92",
    "numpy": "2.4.5",
    "pandas": "3.0.3",
    "scipy": "1.17.1",
    "deep-sort": "1.3.2",
    "hardware": "Apple M2 (MPS) -- FPS das Tabelas 2 e 3",
}

OPENCV_TRACKERS = ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create", "TrackerMedianFlow_create")


def jpeg_size(path: Path) -> tuple[int, int] | None:
    """Le as dimensoes de um JPEG sem decodifica-lo (mesma rotina do pipeline)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from uav_common import jpeg_size as _jpeg_size
    return _jpeg_size(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=Path("datasets/anti_uav_tracking_test_final/images/test"))
    parser.add_argument("--recomendado", action="store_true")
    args = parser.parse_args()

    problems = 0
    print("=" * 74)
    print("AMBIENTE")
    print("=" * 74)
    print(f"  SO            : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python        : {sys.version.split()[0]}  ({sys.executable})")

    try:
        import torch
        print(f"  PyTorch       : {torch.__version__}")
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        suggestion = "mps" if mps else ("0" if cuda else "cpu")
        print(f"  CUDA / MPS    : {cuda} / {mps}      -> use --device {suggestion}")
    except ImportError:
        print("  PyTorch       : AUSENTE -> pip install torch torchvision")
        problems += 1

    try:
        import cv2
        print(f"  OpenCV        : {cv2.__version__}")
        legacy = getattr(cv2, "legacy", None)
        available = []
        for name in OPENCV_TRACKERS:
            if hasattr(cv2, name) or (legacy is not None and hasattr(legacy, name)):
                available.append(name.replace("Tracker", "").replace("_create", ""))
        print(f"  Rastreadores  : {', '.join(available) if available else 'NENHUM'}")
        if "CSRT" not in available:
            print("                  AVISO: sem CSRT. Em Linux use 'opencv-contrib-python';")
            print("                  o pacote 'opencv-python-headless' nao inclui os rastreadores.")
            problems += 1
    except ImportError:
        print("  OpenCV        : AUSENTE -> pip install opencv-contrib-python")
        problems += 1

    for module, package in (("numpy", "numpy"), ("pandas", "pandas"),
                            ("ultralytics", "ultralytics"), ("scipy", "scipy")):
        try:
            imported = __import__(module)
            print(f"  {module:14s}: {getattr(imported, '__version__', 'ok')}")
        except ImportError:
            print(f"  {module:14s}: AUSENTE -> pip install {package}")
            problems += 1

    try:
        from importlib import metadata
        installed = {d.metadata["Name"].lower() for d in metadata.distributions() if d.metadata["Name"]}
        opencv_variants = sorted(n for n in installed if n.startswith("opencv"))
        if len(opencv_variants) > 1:
            print(f"  AVISO OpenCV  : varios pacotes instalados ({', '.join(opencv_variants)}).")
            print("                  Eles se sobrepoem; deixe so o 'opencv-contrib-python':")
            print("                  pip uninstall -y opencv-python opencv-python-headless && "
                  "pip install --force-reinstall opencv-contrib-python")
    except Exception:  # noqa: BLE001
        pass

    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort  # noqa: F401
        print("  deep-sort     : instalado (baseline DeepSORT disponivel)")
    except ImportError as error:
        if "pkg_resources" in str(error):
            print("  deep-sort     : instalado, mas falta pkg_resources -> pip install 'setuptools<81'")
        else:
            print("  deep-sort     : ausente (opcional; pip install deep-sort-realtime)")

    print()
    print("=" * 74)
    print("DATASET")
    print("=" * 74)
    images_dir = args.images_dir.resolve()
    if images_dir.is_dir():
        frames = sorted(images_dir.glob("video*_*.jpg"))
        videos = sorted({p.name.split("_")[0] for p in frames})
        print(f"  Quadros de tracking: {len(frames)} em {len(videos)} sequencias ({images_dir})")
        if frames:
            size = jpeg_size(frames[0])
            print(f"  Resolucao do 1o quadro: {size[0]}x{size[1]}" if size else "  Resolucao: nao foi possivel ler")
        labels = images_dir.parent.parent / "labels" / "test"
        if labels.is_dir():
            print(f"  Rotulos YOLO: {len(list(labels.glob('*.txt')))}")
    else:
        print(f"  AUSENTE: {images_dir}  -> rode o PASSO 2 (src/02_converter_dataset.py --only tracking)")
        problems += 1

    print()
    print("=" * 74)
    print("PESOS")
    print("=" * 74)
    weights_found = sorted(Path("pesos_treinados").rglob("*.pt"))
    for weights in weights_found:
        origin = "treinado por voce" if "treinados_localmente" in weights.parts else "artigo"
        print(f"  {weights}: {weights.stat().st_size / 1e6:.1f} MB ({origin})")
    if not weights_found:
        print("  nenhum peso em pesos_treinados/ -> o repositorio traz yolo26n_dut_best.pt;")
        print("  se ele sumiu, baixe-o de novo do GitHub ou treine no PASSO 3")
        problems += 1

    if args.recomendado:
        print()
        print("=" * 74)
        print("AMBIENTE DO AUTOR (ARTIGO)")
        print("=" * 74)
        for key, value in RECOMMENDED.items():
            print(f"  {key:15s}: {value}")

    print()
    print("=" * 74)
    print("PROBLEMAS BLOQUEANTES: %d" % problems)
    print("=" * 74)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

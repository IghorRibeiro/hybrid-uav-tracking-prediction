#!/usr/bin/env python3
"""OPCIONAL - Rodar OSTrack e ODTrack (baselines Transformer da Tabela 3).

Estes rastreadores NAO fazem parte do metodo proposto: sao baselines de alvo
unico, com repositorio, pesos e ambiente proprios. Este script e a adaptacao do
runner usado no artigo. Ele carrega o rastreador direto do repositorio clonado,
inicializa com a caixa do PRIMEIRO QUADRO do ground truth oficial (como descrito
no artigo) e grava as predicoes no mesmo formato dos demais metodos:

    results/predictions/<ostrack|odtrack>/videoNN.txt   frame,x,y,w,h,conf,tempo_ms

Preparacao (uma vez):
    git clone https://github.com/botaoye/OSTrack.git         externos/OSTrack
    git clone https://github.com/GXNU-ZhongLab/ODTrack.git   externos/ODTrack
    # baixe os pesos indicados no README de cada repositorio e coloque em
    #   externos/OSTrack/output/checkpoints/train/ostrack/<param>/<param>.pth.tar
    #   externos/ODTrack/output/checkpoints/train/odtrack/<param>/<param>.pth.tar
    # instale as dependencias listadas nesses repositorios (timm, easydict, ...)

Pesos usados no artigo:
    OSTrack: vitb_384_mae_ce_32x4_ep300
    ODTrack: baseline

Exemplos:
    python src/extras/rodar_ostrack_odtrack.py --tracker ostrack \\
        --repo externos/OSTrack --param vitb_384_mae_ce_32x4_ep300 --gt-dir datasets/_bruto
    python src/extras/rodar_ostrack_odtrack.py --tracker odtrack \\
        --repo externos/ODTrack --param baseline --gt-dir datasets/_bruto --video video01

Observacao: o codigo desses repositorios foi escrito para CUDA. Para rodar em
Apple Silicon (mps) ou CPU, este script redireciona as chamadas .cuda() para o
dispositivo escolhido e substitui dependencias opcionais (jpeg4py, visdom).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import types
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_common import (  # noqa: E402
    frame_number,
    frames_of_video,
    list_videos,
    read_official_gt,
    read_yolo_labels,
    require_official_gt,
)

TRACKERS = {
    # nome: (modulo, classe, subpasta de config/checkpoint, construtor pede dataset_name?)
    "ostrack": ("lib.test.tracker.ostrack", "OSTrack", "ostrack", True),
    "odtrack": ("lib.test.tracker.odtrack", "ODTrack", "odtrack", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracker", choices=tuple(TRACKERS), required=True)
    parser.add_argument("--repo", type=Path, required=True, help="Repositorio clonado (OSTrack ou ODTrack).")
    parser.add_argument("--param", required=True,
                        help="Nome do yaml em experiments/<tracker>/ e do checkpoint (sem extensao).")
    parser.add_argument("--images-dir", type=Path, default=Path("datasets/anti_uav_tracking_test_final/images/test"))
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="Pasta com os videoNN_gt.txt oficiais (para a caixa inicial).")
    parser.add_argument("--labels-dir", type=Path, default=Path("datasets/anti_uav_tracking_test_final/labels/test"),
                        help="Alternativa ao --gt-dir: rotulos YOLO do PASSO 2.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Padrao: results/predictions/<tracker>")
    parser.add_argument("--video", default=None, help="Processar apenas uma sequencia (ex.: video01).")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps ou cpu.")
    return parser.parse_args()


def resolve_device(requested: str):
    import torch
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def install_compat_shims(device) -> None:
    """Substitui dependencias opcionais e redireciona .cuda() para o dispositivo."""
    import torch

    if not hasattr(torch, "_six"):  # removido em PyTorch recente; alguns repos ainda importam
        six = types.ModuleType("torch._six")
        six.string_classes = (str, bytes)
        torch._six = six
        sys.modules["torch._six"] = six

    if "jpeg4py" not in sys.modules:
        jpeg4py = types.ModuleType("jpeg4py")

        class JPEG:
            def __init__(self, filename):
                self.filename = filename

            def decode(self):
                image = cv2.imread(self.filename)
                if image is None:
                    raise FileNotFoundError(self.filename)
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        jpeg4py.JPEG = JPEG
        sys.modules["jpeg4py"] = jpeg4py

    if "visdom" not in sys.modules:
        visdom = types.ModuleType("visdom")
        visdom.server = types.ModuleType("visdom.server")

        class Visdom:
            def __init__(self, *args, **kwargs):
                pass

            def check_connection(self):
                return True

            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        visdom.Visdom = Visdom
        sys.modules["visdom"] = visdom
        sys.modules["visdom.server"] = visdom.server

    # Checkpoints oficiais (.pth.tar) guardam mais que tensores; o PyTorch >= 2.6
    # passou a exigir weights_only=True por padrao. Os pesos vem dos autores.
    original_load = torch.load

    def load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)
    torch.load = load

    if device.type != "cuda":
        torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(device)
        torch.nn.Module.cuda = lambda self, *args, **kwargs: self.to(device)


def build_tracker(args, device):
    module_name, class_name, subdir, needs_dataset_name = TRACKERS[args.tracker]
    repo = args.repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        tracker_class = getattr(importlib.import_module(module_name), class_name)
        config_module = importlib.import_module(f"lib.config.{subdir}.config")
        from lib.test.tracker.data_utils import Preprocessor
    except (ImportError, AttributeError) as error:
        raise SystemExit(f"Nao consegui importar {class_name} de {repo}: {error}\n"
                         "Confira o clone e as dependencias do repositorio.") from error

    import torch

    def preprocessor_init(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))
    Preprocessor.__init__ = preprocessor_init

    cfg = config_module.cfg
    config_file = repo / "experiments" / subdir / f"{args.param}.yaml"
    if not config_file.is_file():
        raise SystemExit(f"Config nao encontrado: {config_file}")
    config_module.update_config_from_file(str(config_file))

    class Params:
        pass

    params = Params()
    params.cfg = cfg
    params.checkpoint = str(repo / "output" / "checkpoints" / "train" / subdir / args.param
                            / f"{args.param}.pth.tar")
    params.debug = 0
    params.save_all_boxes = False
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    if not Path(params.checkpoint).is_file():
        raise SystemExit(f"Checkpoint nao encontrado: {params.checkpoint}\n"
                         "Baixe os pesos indicados no README do repositorio.")
    return tracker_class(params, "trackingnet") if needs_dataset_name else tracker_class(params)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    install_compat_shims(device)

    images_dir = args.images_dir.resolve()
    output_dir = (args.output_dir or Path("results/predictions") / args.tracker).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = [args.video] if args.video else list_videos(images_dir)
    if not videos:
        raise SystemExit(f"Nenhuma sequencia em {images_dir}. Rode o PASSO 2 primeiro.")

    gt_files = require_official_gt(args.gt_dir) if args.gt_dir else None
    print(f"{args.tracker} ({args.param}) em {device}; {len(videos)} sequencias -> {output_dir}")

    for video_id in videos:
        if gt_files is not None:
            gt = read_official_gt(gt_files[video_id]) if video_id in gt_files else {}
        else:
            gt = read_yolo_labels(args.labels_dir.resolve(), video_id)
        paths = frames_of_video(images_dir, video_id)
        # Caixa inicial: o primeiro quadro anotado com alvo (no DUT, sempre o quadro 1).
        start = next((i for i, p in enumerate(paths) if gt.get(frame_number(p)) is not None), None)
        if start is None:
            print(f"  AVISO {video_id}: sem quadro com alvo no ground truth; pulado.")
            continue
        init_box = list(gt[frame_number(paths[start])])

        tracker = build_tracker(args, device)
        rows = []
        for index, path in enumerate(paths[start:]):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Nao consegui ler {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # como no pipeline oficial dos repos
            tic = time.perf_counter()
            if index == 0:
                tracker.initialize(image, {"init_bbox": init_box})
                box, confidence = init_box, 1.0
            else:
                output = tracker.track(image)
                box, confidence = output["target_bbox"], float(output.get("conf_score", 1.0))
            elapsed_ms = (time.perf_counter() - tic) * 1000.0
            x, y, w, h = (float(v) for v in box)
            rows.append(f"{frame_number(path)},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{confidence:.6f},{elapsed_ms:.3f}")
        (output_dir / f"{video_id}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"  {video_id}: {len(rows)} predicoes")

    print("Proximo passo: PASSO 5, incluindo "
          f"{'OSTrack' if args.tracker == 'ostrack' else 'ODTrack'}={output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PASSO 3 (opcional) - Treinar o detector YOLO26 no conjunto de deteccao do DUT Anti-UAV.

O repositorio ja traz os pesos do YOLO26n usados no artigo em
pesos_treinados/yolo26n_dut_best.pt; este passo so e necessario para quem quer
refazer o ajuste fino (ou treinar o YOLO26m da Tabela 1). Para nao sobrescrever
os pesos do artigo, o melhor peso treinado aqui e copiado para
pesos_treinados/treinados_localmente/<arquitetura>_dut_best.pt.

O artigo usa o YOLO26n; o YOLO26m entra apenas na Tabela 1 (comparacao de
detectores). Os mesmos hiperparametros que produziram os pesos publicados:

    epochs    = 50        imgsz  = 640      batch  = 8
    optimizer = auto      patience = 10     amp    = False (MPS)
    data      = datasets/dut_detection_yolo/dataset.yaml   (gerado no PASSO 2)

Nota sobre o MPS (Apple Silicon): com amp=True o treino trava em alguns
conjuntos de versoes do PyTorch; por isso o padrao aqui e amp=False, como no
treino original. Em CUDA, use --amp para acelerar.

Exemplos:
    python src/03_treinar_detector.py --model yolo26n.pt --name yolo26_n
    python src/03_treinar_detector.py --model yolo26m.pt --name yolo26_m --device 0
    python src/03_treinar_detector.py --data datasets/dut_detection_yolo/dataset.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("datasets/dut_detection_yolo/dataset.yaml"))
    parser.add_argument("--model", default="yolo26n.pt",
                        help="Checkpoint de partida (baixado automaticamente se ausente).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default=None, help="mps, cpu, 0, 0,1 ... (padrao: automatico)")
    parser.add_argument("--name", default="yolo26_n")
    parser.add_argument("--project", type=Path, default=Path("runs/fine_tuning_dut"))
    parser.add_argument("--amp", action="store_true", help="Ligar mixed precision (nao use com MPS).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights-out", type=Path, default=Path("pesos_treinados/treinados_localmente"),
                        help="Onde copiar o best.pt ao final (nao sobrescreve os pesos do artigo).")
    return parser.parse_args()


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()

    data_path = args.data.resolve()
    if not data_path.exists():
        raise SystemExit(
            f"dataset.yaml nao encontrado em {data_path}.\n"
            "Rode primeiro o PASSO 2: python src/02_converter_dataset.py --only detection"
        )

    device = pick_device(args.device)
    print(f"Dispositivo: {device}    AMP: {bool(args.amp)}")
    print(f"Dados:       {data_path}")
    print(f"Modelo base: {args.model}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        amp=bool(args.amp),
        cache=False,
        pretrained=True,
        optimizer="auto",
        patience=args.patience,
        save_period=10,
        seed=args.seed,
        deterministic=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        val=True,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Treino terminou mas {best} nao existe; verifique o log acima.")

    args.weights_out.mkdir(parents=True, exist_ok=True)
    # Os passos 4 e 5 procuram pesos_treinados/<arquitetura>_dut_best.pt.
    architecture = Path(args.model).stem
    destination = args.weights_out / f"{architecture}_dut_best.pt"
    shutil.copy2(best, destination)
    print(f"\nOK melhor peso copiado para {destination.resolve()}")
    print(f"Proximo passo: PASSO 4 com --model {destination}")


if __name__ == "__main__":
    main()

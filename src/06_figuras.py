#!/usr/bin/env python3
"""PASSO 6 - Gerar as figuras a partir das tabelas do PASSO 5.

Todas as figuras sao geradas a partir de resultados calculados (CSV do PASSO 5),
nunca de valores digitados a mao. Quem roda o pipeline obtem as figuras com os
proprios numeros.

    fig3_kalman_damped.png|pdf     Figura 3 do artigo: erro medio de predicao
                                   (H5..H30) do Kalman persistente x Kalman
                                   Damped, barras = desvio entre sequencias.
                                   Le <--ablacao>/ablacao_damping.csv.
    fig_erro_por_horizonte.png     erro H5..H30 de cada metodo (Kalman persistente)
    fig_variabilidade.png          SR@0.5 por sequencia (cada ponto = 1 video)
    fig_sucesso_por_limiar.png     curva de sucesso (SR x limiar de IoU); exige --metodos

Exemplos:
    python src/06_figuras.py --tabelas results/tabela3 --ablacao results/figura3 \
        --figuras results/figures --gt-dir datasets/_bruto \
        --metodos Hibrido=results/predictions/hibrido BoT-SORT=results/predictions/botsort
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uav_common import (
    HORIZONS,
    IOU_THRESHOLDS,
    iou,
    read_official_gt,
    read_predictions,
    read_yolo_labels,
    require_official_gt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tabelas", type=Path, default=Path("results/tabela3"),
                        help="Saida do PASSO 5 com resumo.csv e por_sequencia.csv.")
    parser.add_argument("--ablacao", type=Path, default=None,
                        help="Saida do PASSO 5 com ablacao_damping.csv (padrao: a mesma de --tabelas).")
    parser.add_argument("--figuras", type=Path, default=Path("results/figures"))
    parser.add_argument("--metodos", nargs="*", default=None,
                        help="Nome=pasta (necessario para a curva de sucesso).")
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=Path("datasets/anti_uav_tracking_test_final/labels/test"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def success_curves(args) -> dict:
    """SR medio por limiar de IoU, agregando os quadros de todas as sequencias."""
    # O zip oficial extrai para uma subpasta; localiza os videoNN_gt.txt onde estiverem.
    gt_files = require_official_gt(args.gt_dir) if args.gt_dir else None
    labels_dir = args.labels_dir.resolve()
    curves = {}
    for item in args.metodos:
        name, directory = item.split("=", 1)
        directory = Path(directory).expanduser()
        overlaps = []
        for path in sorted(directory.glob("video*.txt")):
            video_id = path.stem
            if gt_files is not None:
                gt_frames = read_official_gt(gt_files[video_id]) if video_id in gt_files else {}
            else:
                gt_frames = read_yolo_labels(labels_dir, video_id)
            if not gt_frames:
                continue
            predictions = read_predictions(path)
            for frame, box in gt_frames.items():
                if box is None:
                    continue
                prediction = predictions.get(frame)
                overlaps.append(iou(box, prediction[:4]) if prediction is not None else 0.0)
        if not overlaps:
            continue
        overlaps = np.asarray(overlaps)
        curves[name] = [float((overlaps >= t).mean()) for t in IOU_THRESHOLDS]
        print(f"  curva de sucesso: {name} ({overlaps.size} quadros com alvo)")
    return curves


def figure_success(curves: dict, output: Path, dpi: int) -> None:
    plt.figure(figsize=(7, 4.6))
    for name, values in curves.items():
        plt.plot(IOU_THRESHOLDS, values, linewidth=1.8, label=name)
    plt.axvline(0.5, color="grey", linestyle=":", linewidth=1.2)
    plt.text(0.505, 0.05, "IoU = 0,5", color="grey", fontsize=9)
    plt.xlabel("Limiar de IoU")
    plt.ylabel("Taxa de sucesso (SR)")
    plt.title("Curva de sucesso por metodo (todas as sequencias)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    for suffix in (".png", ".pdf"):
        plt.savefig(output.with_suffix(suffix), dpi=dpi)
    plt.close()
    print(f"  figura: {output.with_suffix('.png')}")


def figure_error_horizon(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    plt.figure(figsize=(7, 4.6))
    for _, row in summary.iterrows():
        means = [row[f"H{h}_media"] for h in HORIZONS]
        stds = [row[f"H{h}_std"] for h in HORIZONS]
        plt.errorbar(HORIZONS, means, yerr=stds, fmt="-o", capsize=5, linewidth=1.8, label=row["metodo"])
    plt.xlabel("Horizonte de predicao (quadros a frente)")
    plt.ylabel("Erro medio de posicao (px)")
    plt.title("Erro de predicao de trajetoria por horizonte")
    plt.xticks(list(HORIZONS), [f"H{h}" for h in HORIZONS])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    for suffix in (".png", ".pdf"):
        plt.savefig(output.with_suffix(suffix), dpi=dpi)
    plt.close()
    print(f"  figura: {output.with_suffix('.png')}")


def figure_variability(per_sequence: pd.DataFrame, output: Path, dpi: int) -> None:
    methods = list(dict.fromkeys(per_sequence["metodo"]))
    data = [per_sequence.loc[per_sequence["metodo"] == m, "SR@0.5"].to_numpy(dtype=float) for m in methods]
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    positions = np.arange(1, len(methods) + 1)
    axis.boxplot(data, positions=positions, widths=0.5, showmeans=True)
    for index, values in enumerate(data):
        axis.scatter(np.full(values.size, positions[index]) + np.random.default_rng(0).normal(0, 0.045, values.size),
                     values, s=14, alpha=0.65, zorder=3)
    axis.set_xticks(positions)
    axis.set_xticklabels(methods, rotation=12)
    axis.set_ylabel("SR@0.5 por sequencia")
    axis.set_title(f"Variabilidade entre as {per_sequence['video'].nunique()} sequencias (cada ponto = 1 video)")
    axis.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(output.with_suffix(suffix), dpi=dpi)
    figure.clear()
    plt.close(figure)
    print(f"  figura: {output.with_suffix('.png')}")


def figure_paper_fig3(ablation_path: Path, output: Path, dpi: int) -> None:
    """Figura 3 do artigo: barras agrupadas persistente x damped, por horizonte."""
    table = pd.read_csv(ablation_path)
    for method, group in table.groupby("metodo", sort=False):
        persistent = group[group["damping"] == 1.0]
        damped = group[group["damping"] != 1.0].sort_values("damping", ascending=False)
        if persistent.empty or damped.empty:
            continue
        persistent, damped = persistent.iloc[0], damped.iloc[0]
        positions = np.arange(len(HORIZONS))
        width = 0.36
        figure, axis = plt.subplots(figsize=(7, 4.2))
        for offset, row, color, label in (
            (-width / 2, persistent, "#c0504d", "Kalman Persistente (sem amortecimento)"),
            (width / 2, damped, "#4f81bd", f"Kalman Damped (gamma = {damped['damping']:g})"),
        ):
            means = [row[f"H{h}_media_px"] for h in HORIZONS]
            stds = [row.get(f"H{h}_std_px", np.nan) for h in HORIZONS]
            axis.bar(positions + offset, means, width, yerr=stds, capsize=4, color=color,
                     edgecolor="black", linewidth=0.6, label=label)
        axis.set_xticks(positions)
        axis.set_xticklabels([f"H{h}" for h in HORIZONS])
        axis.set_xlabel("Horizonte de predicao (quadros)")
        axis.set_ylabel("Erro medio de predicao (px)")
        axis.set_ylim(bottom=0)
        axis.grid(True, axis="y", linestyle=":", alpha=0.6)
        axis.legend(frameon=True, fontsize=8, loc="upper left")
        figure.tight_layout()
        suffix_name = output if table["metodo"].nunique() == 1 else output.with_name(f"{output.name}_{method}")
        for suffix in (".png", ".pdf"):
            figure.savefig(suffix_name.with_suffix(suffix), dpi=dpi)
        plt.close(figure)
        reduction = (1 - damped["H30_media_px"] / persistent["H30_media_px"]) * 100
        print(f"  figura: {suffix_name.with_suffix('.png')}  "
              f"(H30 {persistent['H30_media_px']:.1f} -> {damped['H30_media_px']:.1f} px, {reduction:.1f}%)")


def figure_damping(ablation_path: Path, output: Path, dpi: int) -> None:
    """Curvas de erro por horizonte para cada fator de amortecimento (complementa a Figura 3)."""
    table = pd.read_csv(ablation_path)
    figure, axis = plt.subplots(figsize=(7, 4.6))
    for method, group in table.groupby("metodo", sort=False):
        for _, row in group.iterrows():
            means = [row[f"H{h}_media_px"] for h in HORIZONS]
            damping = row["damping"]
            label = f"{method} damping={damping:g}" if damping != 1.0 else f"{method} persistente"
            linestyle = "-" if damping == 1.0 else "--"
            marker = "o" if damping == 1.0 else "s"
            axis.plot(HORIZONS, means, linestyle=linestyle, marker=marker, linewidth=1.8,
                      label=label)
    axis.set_xlabel("Horizonte de predicao (quadros a frente)")
    axis.set_ylabel("Erro medio de posicao (px)")
    axis.set_title("Efeito do amortecimento na predicao de trajetoria")
    axis.set_xticks(list(HORIZONS))
    axis.set_xticklabels([f"H{h}" for h in HORIZONS])
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.legend(frameon=True, fontsize=9)
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(output.with_suffix(suffix), dpi=dpi)
    figure.clear()
    plt.close(figure)
    print(f"  figura: {output.with_suffix('.png')}")


def main() -> None:
    args = parse_args()
    tables = args.tabelas.resolve()
    figures = args.figuras.resolve()
    figures.mkdir(parents=True, exist_ok=True)

    summary_path = tables / "resumo.csv"
    per_sequence_path = tables / "por_sequencia.csv"
    ablation_path = (args.ablacao.resolve() if args.ablacao else tables) / "ablacao_damping.csv"
    if not summary_path.exists() and not ablation_path.exists():
        raise SystemExit(f"Tabelas nao encontradas em {tables}. Rode o PASSO 5 primeiro.")

    print("Gerando figuras ...")
    if ablation_path.exists():
        figure_paper_fig3(ablation_path, figures / "fig3_kalman_damped", args.dpi)
        figure_damping(ablation_path, figures / "fig_damping_curvas", args.dpi)
    else:
        print("  (Figura 3 ignorada: rode o PASSO 5 com --ablar-damping)")

    if summary_path.exists() and per_sequence_path.exists():
        summary = pd.read_csv(summary_path)
        per_sequence = pd.read_csv(per_sequence_path)
        figure_error_horizon(summary, figures / "fig_erro_por_horizonte", args.dpi)
        figure_variability(per_sequence, figures / "fig_variabilidade", args.dpi)

    if args.metodos:
        curves = success_curves(args)
        if curves:
            figure_success(curves, figures / "fig_sucesso_por_limiar", args.dpi)
    else:
        print("  (curva de sucesso ignorada: passe --metodos Nome=pasta para gera-la)")

    print(f"\nFiguras gravadas em {figures}")


if __name__ == "__main__":
    main()

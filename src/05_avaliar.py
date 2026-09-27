#!/usr/bin/env python3
"""PASSO 5 - Avaliar as predicoes e montar as tabelas do artigo.

Le as predicoes do PASSO 4 (results/predictions/<metodo>/videoNN.txt) e o
ground truth oficial, e escreve em --output-dir:

    por_sequencia.csv     uma linha por (metodo, sequencia), todas as metricas
    resumo.csv            media e desvio entre as 20 sequencias, IC95% (t de
                          Student) do H30 e IC95% (bootstrap) do SR@0.5
    comparacao.csv        formato das Tabelas 2 e 3 do artigo: SA, SR@0.5, IoU,
                          AUC, H30 (media +/- desvio), ganho de H30 sobre a
                          referencia e FPS
    testes_pareados.csv   t pareado e Wilcoxon por sequencia contra a referencia
    ablacao_damping.csv   (com --ablar-damping) erro H5..H30 por fator de
                          amortecimento -- a Figura 3 do artigo
    tabela1_detectores.csv (com --tabela1) precisao, recall, mAP do YOLO26n/m

Metricas (definicoes em src/uav_common.py, fonte unica para todos os passos):
    SA       [quadros com alvo e IoU>=0,5] + [quadros sem alvo corretamente
             declarados sem alvo], dividido pelo total de quadros
    SR@0.5   fracao dos quadros COM alvo em que IoU >= 0,5
    IoU      media do IoU nos quadros com alvo (sem predicao conta como 0)
    AUC      area sob a curva de sucesso (limiares de IoU 0..1, passo 0,05)
    H5..H30  erro de predicao de trajetoria (px) com Kalman CV
    FPS      quadros por segundo do rastreador, medidos no SEU dispositivo

Exemplos:
    # Tabela 3 (a referencia do ganho de H30 e o BoT-SORT, como no artigo)
    python src/05_avaliar.py --gt-dir datasets/_bruto --output-dir results/tabela3 \\
        --referencia BoT-SORT \\
        --metodos BoT-SORT=results/predictions/botsort \\
                  ByteTrack=results/predictions/bytetrack \\
                  Hibrido=results/predictions/hibrido

    # Figura 3: Kalman persistente (1,0) contra Kalman Damped (0,95) no hibrido
    python src/05_avaliar.py --gt-dir datasets/_bruto --output-dir results/figura3 \\
        --metodos Hibrido=results/predictions/hibrido --ablar-damping 1.0 0.95

    # Tabela 1 (roda o val() do Ultralytics nas 24.804 imagens de tracking)
    python src/05_avaliar.py --output-dir results/tabela1 --tabela1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from uav_common import (
    HORIZONS,
    centers_from_gt,
    centers_from_predictions,
    ci95_t,
    prediction_errors,
    read_official_gt,
    read_predictions,
    read_yolo_labels,
    require_official_gt,
    summarize_errors,
    tracking_metrics,
)

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metodos", nargs="*", default=[],
                        help="Pares Nome=pasta, por exemplo Hibrido=results/predictions/hibrido")
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="Pasta (ou pasta-mae) com os videoNN_gt.txt oficiais. Recomendado.")
    parser.add_argument("--labels-dir", type=Path, default=Path("datasets/anti_uav_tracking_test_final/labels/test"),
                        help="Alternativa ao GT oficial: rotulos YOLO gerados no PASSO 2.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/avaliacao"))
    parser.add_argument("--damping", type=float, default=1.0,
                        help="Amortecimento do preditor Kalman nas tabelas. 1.0 = Kalman persistente, "
                             "o protocolo comum usado nas Tabelas 2 e 3 do artigo.")
    parser.add_argument("--referencia", default=None,
                        help="Metodo de referencia do ganho de H30 e dos testes pareados "
                             "(padrao: o primeiro de --metodos). No artigo: BoT-SORT.")
    parser.add_argument("--ablar-damping", nargs="*", type=float, default=None, metavar="FATOR",
                        help="Ablacao do amortecimento (Figura 3). Sem valores, usa 1.0 0.95 0.90 0.85.")
    parser.add_argument("--tabela1", action="store_true", help="Rodar tambem a validacao dos detectores (Tabela 1).")
    parser.add_argument("--pesos-yolo26n", type=Path, default=Path("pesos_treinados/yolo26n_dut_best.pt"))
    parser.add_argument("--pesos-yolo26m", type=Path, default=Path("pesos_treinados/yolo26m_dut_best.pt"))
    parser.add_argument("--dados-tabela1", type=Path,
                        default=Path("datasets/anti_uav_tracking_test_final/dataset.yaml"),
                        help="dataset.yaml avaliado na Tabela 1. O artigo usa as 24.804 imagens das "
                             "20 sequencias de tracking (split 'test' gerado no PASSO 2).")
    parser.add_argument("--split-tabela1", default="test")
    parser.add_argument("--device", default="auto", help="Dispositivo do val() da Tabela 1.")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Estatistica
# --------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 2026) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


# --------------------------------------------------------------------------
# Avaliacao
# --------------------------------------------------------------------------

def evaluate_method(name: str, directory: Path, ground_truth, damping: float) -> pd.DataFrame:
    rows = []
    if not directory.is_dir():
        print(f"  AVISO {name}: pasta nao encontrada ({directory}); metodo ignorado.")
        return pd.DataFrame(rows)

    prediction_files = sorted(directory.glob("video*.txt"))
    if not prediction_files:
        print(f"  AVISO {name}: nenhum videoNN.txt em {directory}; metodo ignorado.")
        return pd.DataFrame(rows)

    for path in prediction_files:
        video_id = path.stem
        gt_frames = ground_truth(video_id)
        if not gt_frames:
            print(f"  AVISO {name}/{video_id}: sem ground truth; sequencia ignorada.")
            continue
        predictions = read_predictions(path)
        metrics = tracking_metrics(gt_frames, predictions)
        errors = summarize_errors(
            prediction_errors(centers_from_predictions(predictions), centers_from_gt(gt_frames), damping)
        )
        rows.append({"metodo": name, "video": video_id, **metrics, **errors})
        print(f"  {name}/{video_id}: SR@0.5={metrics['SR@0.5']:.4f}  SA={metrics['SA']:.4f}  "
              f"H30={errors['H30_media']:.2f}px  FPS={metrics['FPS']:.1f}")
    return pd.DataFrame(rows)


def summarize(per_sequence: pd.DataFrame, damping: float) -> pd.DataFrame:
    """Media, desvio e IC95% entre sequencias, por metodo (a unidade e a sequencia)."""
    summary_rows = []
    for name, group in per_sequence.groupby("metodo", sort=False):
        row = {"metodo": name,
               "damping": damping,
               "sequencias": len(group),
               "quadros": int(group["frames_total"].sum()),
               "SA": group["SA"].mean(),
               "SA_std": group["SA"].std(ddof=1),
               "SR@0.5": group["SR@0.5"].mean(),
               "SR@0.5_std": group["SR@0.5"].std(ddof=1),
               "IoU_medio": group["IoU_medio"].mean(),
               "AUC": group["AUC"].mean(),
               "FPS": group["FPS"].mean(),
               "cobertura": group["cobertura"].mean()}
        low, high = bootstrap_ci(group["SR@0.5"].to_numpy(dtype=float))
        row["SR@0.5_IC95_inf"], row["SR@0.5_IC95_sup"] = low, high
        for horizon in HORIZONS:
            values = group[f"H{horizon}_media"].to_numpy(dtype=float)
            row[f"H{horizon}_media"] = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
            row[f"H{horizon}_std"] = float(pd.Series(values).std(ddof=1))
            row[f"H{horizon}_mediana"] = float(np.nanmedian(values)) if np.isfinite(values).any() else math.nan
        row["H30_IC95_inf"], row["H30_IC95_sup"] = ci95_t(group["H30_media"].to_numpy(dtype=float))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def comparison_table(summary: pd.DataFrame, reference_name: str) -> pd.DataFrame:
    """Tabela no formato do artigo, com o ganho percentual de H30 sobre a referencia."""
    reference = summary.loc[summary["metodo"] == reference_name].iloc[0]
    rows = []
    for _, row in summary.iterrows():
        gain = ""
        if row["metodo"] == reference_name:
            gain = "-"
        elif reference["H30_media"] and not math.isnan(reference["H30_media"]):
            gain = f"{(1 - row['H30_media'] / reference['H30_media']) * 100:+.1f}"
        rows.append({
            "Metodo": row["metodo"],
            "SA": round(row["SA"], 3),
            "SR@0.5": round(row["SR@0.5"], 3),
            "IoU_medio": round(row["IoU_medio"], 3),
            "AUC": round(row["AUC"], 3),
            "H30_px": f"{row['H30_media']:.1f} +/- {row['H30_std']:.1f}",
            "H30_IC95_px": f"[{row['H30_IC95_inf']:.1f}; {row['H30_IC95_sup']:.1f}]",
            f"Ganho_H30_%_vs_{reference_name}": gain,
            "FPS": round(row["FPS"], 1) if not math.isnan(row["FPS"]) else math.nan,
            "Sequencias": int(row["sequencias"]),
        })
    return pd.DataFrame(rows)


def paired_tests(per_sequence: pd.DataFrame, reference_name: str) -> pd.DataFrame:
    """t pareado e Wilcoxon por sequencia, com IC95% da diferenca (bootstrap)."""
    if scipy_stats is None:
        print("  AVISO scipy ausente: testes pareados nao serao calculados (pip install scipy).")
        return pd.DataFrame()

    metrics = ["SR@0.5", "IoU_medio", "AUC", "H5_media", "H10_media", "H15_media", "H30_media", "FPS"]
    reference = per_sequence[per_sequence["metodo"] == reference_name].set_index("video")
    rows = []
    for name, group in per_sequence.groupby("metodo", sort=False):
        if name == reference_name:
            continue
        other = group.set_index("video")
        common = sorted(set(reference.index) & set(other.index))
        if len(common) < 3:
            continue
        for metric in metrics:
            a = other.loc[common, metric].to_numpy(dtype=float)
            b = reference.loc[common, metric].to_numpy(dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            if a.size < 3:
                continue
            difference = a - b
            low, high = bootstrap_ci(difference)
            entry = {
                "metodo": name,
                "referencia": reference_name,
                "metrica": metric,
                "n_sequencias": int(a.size),
                "media_metodo": float(a.mean()),
                "media_referencia": float(b.mean()),
                "diferenca": float(difference.mean()),
                "IC95_inf": low,
                "IC95_sup": high,
                "significativo_95": "sim" if (low > 0 and high > 0) or (low < 0 and high < 0) else "nao",
            }
            entry["p_t_pareado"] = float(scipy_stats.ttest_rel(a, b).pvalue)
            try:
                entry["p_wilcoxon"] = float(scipy_stats.wilcoxon(a, b).pvalue)
            except ValueError:
                entry["p_wilcoxon"] = math.nan
            rows.append(entry)
    return pd.DataFrame(rows)


def damping_ablation(methods: list, ground_truth, dampings: list[float]) -> pd.DataFrame:
    """Erro por horizonte para cada fator de amortecimento (Figura 3 do artigo).

    Le as MESMAS predicoes e recalcula apenas o preditor Kalman: a unica coisa
    que muda entre as linhas e o amortecimento, nao o rastreador.
    """
    rows = []
    for name, directory in methods:
        prediction_files = sorted(directory.glob("video*.txt")) if directory.is_dir() else []
        if not prediction_files:
            print(f"  AVISO ablacao: {name} sem predicoes em {directory}")
            continue
        for damping in dampings:
            per_video = {horizon: [] for horizon in HORIZONS}
            for path in prediction_files:
                gt_frames = ground_truth(path.stem)
                if not gt_frames:
                    continue
                errors = prediction_errors(centers_from_predictions(read_predictions(path)),
                                           centers_from_gt(gt_frames), damping)
                for horizon in HORIZONS:
                    values = errors.get(horizon) or []
                    if values:
                        per_video[horizon].append(float(np.mean(values)))
            row = {"metodo": name, "damping": damping, "sequencias": len(per_video[HORIZONS[0]])}
            for horizon in HORIZONS:
                values = np.asarray(per_video[horizon], dtype=float)
                row[f"H{horizon}_media_px"] = float(values.mean()) if values.size else math.nan
                row[f"H{horizon}_std_px"] = float(values.std(ddof=1)) if values.size > 1 else math.nan
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Reducao percentual em relacao ao persistente DO MESMO metodo, por horizonte.
    for horizon in HORIZONS:
        reductions = []
        for _, row in table.iterrows():
            base = table[(table["metodo"] == row["metodo"]) & (table["damping"] == 1.0)]
            if base.empty or not base.iloc[0][f"H{horizon}_media_px"]:
                reductions.append(math.nan)
                continue
            reductions.append((1 - row[f"H{horizon}_media_px"] / base.iloc[0][f"H{horizon}_media_px"]) * 100)
        table[f"reducao_H{horizon}_pct"] = reductions
    return table


def resolve_device(requested: str) -> str:
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


def run_detector_table(args: argparse.Namespace) -> pd.DataFrame:
    """Tabela 1: validacao dos detectores nas imagens das sequencias de tracking."""
    from ultralytics import YOLO

    data = args.dados_tabela1.resolve()
    if not data.exists():
        print(f"  AVISO Tabela 1 ignorada: {data} nao existe (rode o PASSO 2 --only tracking).")
        return pd.DataFrame()

    rows = []
    for label, weights in (("YOLO26n", args.pesos_yolo26n), ("YOLO26m", args.pesos_yolo26m)):
        weights = weights.resolve()
        if not weights.exists():
            print(f"  AVISO Tabela 1: pesos de {label} nao encontrados ({weights}); linha ignorada.")
            continue
        print(f"  validando {label} em {data} (split {args.split_tabela1}) ...")
        model = YOLO(str(weights))
        metrics = model.val(data=str(data), split=args.split_tabela1,
                            device=resolve_device(args.device), verbose=False)
        rows.append({"Modelo": label,
                     "Precisao": round(float(metrics.box.mp), 3),
                     "Recall": round(float(metrics.box.mr), 3),
                     "mAP@0.5": round(float(metrics.box.map50), 3),
                     "mAP@0.5:0.95": round(float(metrics.box.map), 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.metodos and not args.tabela1:
        raise SystemExit("Nada a fazer: passe --metodos Nome=pasta e/ou --tabela1.")

    detector_table = run_detector_table(args) if args.tabela1 else pd.DataFrame()
    if not detector_table.empty:
        detector_table.to_csv(output_dir / "tabela1_detectores.csv", index=False)
        print("\n" + "=" * 78)
        print("TABELA 1 - DETECTORES YOLO26 (imagens das 20 sequencias do DUT Anti-UAV)")
        print("=" * 78)
        print(detector_table.to_string(index=False))

    if not args.metodos:
        print(f"\nArquivos gravados em {output_dir}")
        return

    # ---- ground truth
    if args.gt_dir is not None:
        # Aceita a pasta direta ou qualquer pasta-mae (o zip oficial extrai para
        # uma subpasta, ex.: _bruto/tracking/gt/Anti-UAV-Tracking-V0GT/).
        gt_files = require_official_gt(args.gt_dir)

        def ground_truth(video_id: str):
            return read_official_gt(gt_files[video_id]) if video_id in gt_files else {}
        source_label = f"GT oficial ({len(gt_files)} sequencias em {args.gt_dir})"
    else:
        labels_dir = args.labels_dir.resolve()

        def ground_truth(video_id: str):
            return read_yolo_labels(labels_dir, video_id)
        source_label = f"rotulos YOLO ({labels_dir})"
    print(f"Ground truth: {source_label}")
    print(f"Amortecimento do preditor nas tabelas (damping): {args.damping}")

    # ---- metodos
    methods = []
    for item in args.metodos:
        if "=" not in item:
            raise SystemExit(f"--metodos espera Nome=pasta; recebi '{item}'")
        name, directory = item.split("=", 1)
        methods.append((name, Path(directory).expanduser()))

    frames = []
    for name, directory in methods:
        print(f"\nAvaliando {name} ({directory})")
        frames.append(evaluate_method(name, directory, ground_truth, args.damping))
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("Nenhuma predicao foi avaliada. Rode o PASSO 4 primeiro.")

    per_sequence = pd.concat(frames, ignore_index=True)
    per_sequence.to_csv(output_dir / "por_sequencia.csv", index=False)

    summary = summarize(per_sequence, args.damping)
    summary.to_csv(output_dir / "resumo.csv", index=False)

    reference = args.referencia or summary.iloc[0]["metodo"]
    if reference not in set(summary["metodo"]):
        raise SystemExit(f"--referencia '{reference}' nao esta entre os metodos avaliados.")
    table = comparison_table(summary, reference)
    table.to_csv(output_dir / "comparacao.csv", index=False)

    tests = paired_tests(per_sequence, reference)
    if not tests.empty:
        tests.to_csv(output_dir / "testes_pareados.csv", index=False)

    if args.ablar_damping is not None:
        factors = args.ablar_damping or [1.0, 0.95, 0.90, 0.85]
        if 1.0 not in factors:
            factors = [1.0, *factors]  # a reducao e sempre contra o persistente
        ablation = damping_ablation(methods, ground_truth, factors)
    else:
        ablation = pd.DataFrame()
    if not ablation.empty:
        ablation.to_csv(output_dir / "ablacao_damping.csv", index=False)

    # ---- saida no terminal
    print("\n" + "=" * 78)
    print(f"COMPARACAO (Kalman damping={args.damping}; ganho de H30 sobre {reference})")
    print("=" * 78)
    print(table.to_string(index=False))

    if not tests.empty:
        print("\n" + "=" * 78)
        print(f"TESTES PAREADOS POR SEQUENCIA (referencia: {reference})")
        print("=" * 78)
        interesting = tests[tests["metrica"].isin(["SR@0.5", "H30_media", "FPS"])]
        print(interesting[["metodo", "metrica", "media_metodo", "media_referencia",
                           "diferenca", "IC95_inf", "IC95_sup", "p_t_pareado",
                           "significativo_95"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not ablation.empty:
        print("\n" + "=" * 78)
        print("ABLACAO DO AMORTECIMENTO (mesmas predicoes; so o preditor Kalman muda)")
        print("=" * 78)
        columns = ["metodo", "damping", *[f"H{h}_media_px" for h in HORIZONS], "H30_std_px", "reducao_H30_pct"]
        print(ablation[columns].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print(f"\nArquivos gravados em {output_dir}")


if __name__ == "__main__":
    main()

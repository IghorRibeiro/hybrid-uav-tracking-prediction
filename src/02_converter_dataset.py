#!/usr/bin/env python3
"""PASSO 2 - Converter o dataset DUT Anti-UAV para o formato YOLO.

Gera duas coisas:

(A) datasets/anti_uav_tracking_test_final/  -- as 20 sequencias de TRACKING no
    formato imagem + rotulo YOLO, que e o que os rastreadores consomem:
        images/test/videoNN_00001.jpg
        labels/test/videoNN_00001.txt   -> "0 cx cy w h" normalizado
    Quadro sem alvo no ground truth oficial ("-100 -100 -100 -100") vira um
    arquivo de rotulo VAZIO -- convencao do YOLO para "nenhum objeto".

(B) datasets/dut_detection_yolo/  -- o conjunto de DETECCAO (10.000 imagens)
    convertido de XML para YOLO, usado no PASSO 3 para treinar o detector:
        images/{train,val,test}/ + labels/{train,val,test}/ + dataset.yaml

Os dois conjuntos vem do mesmo dataset oficial, de modo que o detector
avaliado no PASSO 3 e treinado no MESMO dominio das imagens do PASSO 4.

Exemplos:
    # conversao de tracking (e a que importa para reproduzir o artigo)
    python src/02_converter_dataset.py --raw datasets/_bruto --only tracking

    # conversao de deteccao, para treinar o detector do zero
    python src/02_converter_dataset.py --raw datasets/_bruto --only detection
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml

from uav_common import jpeg_size, official_gt_files, write_resolutions

CLASS_ID = 0
CLASS_NAME = "drone"

# O conjunto de deteccao rotula a classe como "UAV"; o de tracking nao tem
# classe. Unificamos o nome em dataset.yaml e aceitamos os dois rotulos.
DETECTION_CLASS_ALIASES = {"UAV", "uav", "drone", "Drone"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("datasets/_bruto"),
                        help="Diretorio do PASSO 1 com o dataset oficial.")
    parser.add_argument("--tracking-out", type=Path, default=Path("datasets/anti_uav_tracking_test_final"))
    parser.add_argument("--detection-out", type=Path, default=Path("datasets/dut_detection_yolo"))
    parser.add_argument("--only", choices=("all", "tracking", "detection"), default="all")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="Para deteccao: subconjuntos a converter (padrao: train val test).")
    return parser.parse_args()


def _numeric_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.stem.rsplit("_", 1)[-1] if ch.isdigit())
    return (int(digits) if digits else -1, path.name)


def locate_frames(raw: Path, video_id: str) -> list[Path]:
    """Quadros JPEG de uma sequencia de TRACKING, em ordem numerica.

    Aceita as organizacoes conhecidas do material oficial:

        .../Anti-UAV-Tracking-V0/video01/00001.jpg   (pacote oficial de imagens)
        .../img/video01/00001.jpg
        .../video01_00001.jpg                        (pasta plana)

    Pastas do conjunto de DETECCAO sao ignoradas (as imagens de deteccao nao
    pertencem as sequencias de tracking). Arquivos '._*' do macOS tambem.
    """
    def is_tracking(path: Path) -> bool:
        return "detection" not in {part.lower() for part in path.parts}

    directories = sorted(d for d in raw.rglob(video_id) if d.is_dir() and is_tracking(d))
    for directory in directories:
        frames = [p for p in directory.glob("*.jpg") if not p.name.startswith("._")]
        if frames:
            return sorted(frames, key=_numeric_key)
    flat = [p for p in raw.rglob(f"{video_id}_*.jpg") if not p.name.startswith("._") and is_tracking(p)]
    return sorted(flat, key=_numeric_key)


def convert_tracking(gt_files: dict[str, Path], raw: Path, out_dir: Path) -> None:
    images_out = out_dir / "images" / "test"
    labels_out = out_dir / "labels" / "test"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)


    total_frames = 0
    total_absent = 0
    sizes: dict[str, tuple[int, int]] = {}
    extracted = 0
    skipped = []
    print(f"Ground truth: {len(gt_files)} sequencias ({sorted({str(p.parent) for p in gt_files.values()})[0]})")
    print(f"Material:     {raw}")

    for video_id, gt_file in gt_files.items():
        annotations = [line.strip() for line in gt_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        # (1) Quadros ja disponiveis como imagem? (caso do pacote oficial)
        image_paths = locate_frames(raw, video_id)

        # (2) Sem imagens: extrai os quadros do proprio video oficial (.mp4/.avi).
        if not image_paths:
            video_file = find_video_file(raw, video_id)
            if video_file is not None:
                image_paths = extract_frames(video_file, images_out, video_id)
                extracted += len(image_paths)
                if image_paths:
                    print(f"  {video_id}: {len(image_paths)} quadros extraidos de {video_file.name}")

        if not image_paths:
            print(f"  PULADO {video_id}: sem quadros (nem imagens nem video) para esta sequencia.")
            print(f"          baixe o material completo e rode de novo; "
                  f"o ground truth tem {len(annotations)} anotacoes.")
            skipped.append(video_id)
            continue

        if len(image_paths) != len(annotations):
            print(f"  AVISO {video_id}: {len(image_paths)} quadros != {len(annotations)} anotacoes; "
                  "usando o menor (os quadros devem comecar em 1 e ser consecutivos).")
        pairs = min(len(image_paths), len(annotations))

        # (3) Resolucao REAL desta sequencia (elas nao sao todas iguais).
        size = jpeg_size(image_paths[0])
        if size is None:
            frame = cv2.imread(str(image_paths[0]))
            size = (frame.shape[1], frame.shape[0]) if frame is not None else None
        if size is None:
            print(f"  PULADO {video_id}: nao consegui ler a resolucao do primeiro quadro.")
            skipped.append(video_id)
            continue
        width, height = size
        sizes[video_id] = (width, height)

        for position in range(pairs):
            image_path = image_paths[position]
            raw_annotation = annotations[position]
            # Nome padronizado: videoNN_00001.jpg. A linha i do ground truth
            # corresponde ao i-esimo quadro da sequencia (i comeca em 1).
            out_name = f"{video_id}_{position + 1:05d}.jpg"
            target = images_out / out_name
            if image_path.resolve() != target.resolve():
                shutil.copy2(image_path, target)
            write_label(labels_out / Path(out_name).with_suffix(".txt"), raw_annotation, width, height)
            total_frames += 1
            if raw_annotation.startswith("-100"):
                total_absent += 1
        print(f"  {video_id}: {pairs} quadros em {width}x{height}")

    if sizes:
        write_resolutions(out_dir / "resolutions.json", sizes)
        different = {v: s for v, s in sizes.items() if s != (1920, 1080)}
        print(f"\nResolucoes gravadas em {out_dir / 'resolutions.json'}")
        if different:
            print(f"  atencao: {len(different)} sequencia(s) NAO sao 1920x1080 -> "
                  f"{', '.join(f'{v}={w}x{h}' for v, (w, h) in list(different.items())[:8])}")

    write_dataset_yaml(out_dir, out_dir / "images" / "test")
    if extracted:
        print(f"({extracted} quadros foram extraidos de videos pelo proprio pipeline)")
    print(f"\nOK tracking: {len(sizes)} sequencias, {total_frames} quadros "
          f"({total_absent} sem alvo) em {out_dir}")
    if total_frames != 24804 or skipped:
        print(f"  ATENCAO: o artigo usa 20 sequencias / 24.804 quadros. "
              f"Sequencias puladas: {', '.join(skipped) or 'nenhuma'}.")


def find_video_file(raw: Path, video_id: str) -> Path | None:
    """Localiza o video oficial da sequencia (videoNN.mp4 / .avi) em qualquer lugar do material baixado."""
    for suffix in (".mp4", ".avi", ".mov", ".mkv"):
        matches = sorted(raw.rglob(f"{video_id}{suffix}")) + sorted(raw.rglob(f"*{video_id}*{suffix}"))
        matches = [m for m in matches if not m.name.startswith("._")]
        if matches:
            return matches[0]
    return None


def extract_frames(video_file: Path, images_out: Path, video_id: str) -> list[Path]:
    """Extrai TODOS os quadros do video para JPEG, numerados a partir de 1."""
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        print(f"  AVISO nao consegui abrir {video_file}")
        return []
    paths: list[Path] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        out_path = images_out / f"{video_id}_{index:05d}.jpg"
        cv2.imwrite(str(out_path), frame)
        paths.append(out_path)
    capture.release()
    return paths


def write_label(label_path: Path, annotation: str, width: int, height: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    parts = annotation.split()
    if len(parts) != 4:
        label_path.write_text("", encoding="utf-8")
        return
    x, y, w, h = (float(v) for v in parts)
    if x < 0 or y < 0 or w < 0 or h < 0:
        label_path.write_text("", encoding="utf-8")
        return
    cx = min(1.0, max(0.0, (x + w / 2.0) / width))
    cy = min(1.0, max(0.0, (y + h / 2.0) / height))
    nw = min(1.0, max(0.0, w / width))
    nh = min(1.0, max(0.0, h / height))
    label_path.write_text(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n", encoding="utf-8")


def convert_detection(raw: Path, out_dir: Path, splits: list[str]) -> None:
    """Converte as anotacoes XML do conjunto de deteccao para o formato YOLO."""
    converted_splits = []
    for split in splits:
        img_dir = xml_dir = None
        for candidate in (raw / "detection" / split, raw / split, raw / "detection"):
            if not candidate.exists():
                continue
            img_dir = img_dir or next((d for d in (candidate / "img", candidate / "images", candidate) if d.is_dir() and list(d.glob("*.jpg"))), None)
            xml_dir = xml_dir or next((d for d in (candidate / "xml", candidate / "annotations", candidate) if d.is_dir() and list(d.glob("*.xml"))), None)
        if img_dir is None or xml_dir is None:
            print(f"  AVISO {split}: nao encontrei img/ e xml/ (procurado em {raw}). Pulando.")
            continue

        images_out = out_dir / "images" / split
        labels_out = out_dir / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        xml_files = sorted(xml_dir.glob("*.xml"))
        count = 0
        boxes = 0
        unreadable = 0      # XML malformado ou ilegivel
        missing_image = 0   # anotacao sem a imagem correspondente
        for xml_file in xml_files:
            try:
                root = ET.parse(xml_file).getroot()
            except ET.ParseError as error:
                unreadable += 1
                if unreadable <= 3:
                    print(f"  AVISO XML ilegivel: {xml_file.name} ({error})")
                continue
            filename = (root.findtext("filename") or xml_file.stem + ".jpg").strip()
            source_image = img_dir / filename
            if not source_image.exists():
                missing_image += 1
                if missing_image <= 3:
                    print(f"  AVISO sem imagem para {xml_file.name}: procurei {filename}")
                continue
            size = root.find("size")
            width = int(size.findtext("width")) if size is not None and size.findtext("width") else 1920
            height = int(size.findtext("height")) if size is not None and size.findtext("height") else 1080

            lines = []
            for obj in root.findall("object"):
                name = (obj.findtext("name") or "").strip()
                if name not in DETECTION_CLASS_ALIASES:
                    continue
                box = obj.find("bndbox")
                if box is None:
                    continue
                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
                cx = min(1.0, max(0.0, (xmin + xmax) / 2.0 / width))
                cy = min(1.0, max(0.0, (ymin + ymax) / 2.0 / height))
                nw = min(1.0, max(0.0, (xmax - xmin) / width))
                nh = min(1.0, max(0.0, (ymax - ymin) / height))
                lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                boxes += 1

            shutil.copy2(source_image, images_out / filename)
            (labels_out / Path(filename).with_suffix(".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            count += 1
        print(f"  {split}: {count} imagens, {boxes} caixas")
        if unreadable or missing_image:
            print(f"  ATENCAO {split}: {unreadable} XML ilegivel(is) e "
                  f"{missing_image} anotacao(oes) sem imagem foram descartadas.")
            print("          Se o numero de imagens ficou muito abaixo do esperado, "
                  "o material baixado esta incompleto ou corrompido.")
        if count == 0:
            # Nao adianta registrar o split: um conjunto vazio faz o treino falhar
            # com um erro obscuro do Ultralytics ("No images found").
            print(f"  ERRO {split}: nenhuma imagem convertida; split NAO sera usado.")
            continue
        converted_splits.append(split)

    if converted_splits:
        write_dataset_yaml(out_dir, None, converted_splits)
        print(f"\nOK deteccao: {out_dir} (splits: {', '.join(converted_splits)})")


def write_dataset_yaml(out_dir: Path, test_dir: Path | None, converted_splits: list[str] | None = None) -> None:
    """Escreve um dataset.yaml com caminhos ABSOLUTOS (exigencia do Ultralytics)."""
    out_dir = out_dir.resolve()
    if converted_splits is not None:
        train = "images/train" if "train" in converted_splits else ""
        val = "images/val" if "val" in converted_splits else ""
        test = "images/test" if "test" in converted_splits else ""
    else:
        train = val = ""
        test = "images/test"
        _ = test_dir
    config = {"path": str(out_dir), "train": train, "val": val, "test": test,
              "nc": 1, "names": [CLASS_NAME]}
    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    raw = args.raw.resolve()

    if args.only in ("all", "tracking"):
        gt_files = official_gt_files(raw)
        if not gt_files:
            raise SystemExit(f"ERRO: nao encontrei videoNN_gt.txt em {raw}.\n"
                             "Rode o PASSO 1 (src/01_baixar_dataset.py) ou passe --raw apontando\n"
                             "para a pasta com o dataset oficial.")
        convert_tracking(gt_files, raw, args.tracking_out.resolve())

    if args.only in ("all", "detection"):
        splits = args.splits or ["train", "val", "test"]
        convert_detection(raw, args.detection_out.resolve(), splits)


if __name__ == "__main__":
    main()

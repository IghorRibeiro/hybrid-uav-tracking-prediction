#!/usr/bin/env python3
"""PASSO 1 - Obter o dataset DUT Anti-UAV na origem oficial.

Nada neste pipeline depende de arquivos baixados de terceiros: o dataset vem
do repositorio oficial dos autores do DUT Anti-UAV.

    Jie Zhao, Jingshu Zhang, Dongdong Li, Dong Wang.
    "Vision-based Anti-UAV Detection and Tracking".
    IEEE Transactions on Intelligent Transportation Systems, 2022.
    https://github.com/wangdongdut/DUT-Anti-UAV
    https://arxiv.org/abs/2205.10851

O dataset tem DUAS partes independentes:

  Detection (10.000 imagens + anotacoes XML)  -> usada no PASSO 3 (treino do YOLO)
  Tracking  (20 sequencias de video + ground truth) -> usada no PASSO 4 (rastreamento)

Este script baixa as duas para --dest (padrao: datasets/_bruto) e confere a
estrutura. Os links sao os do README oficial; se o Google Drive mudar de
politica de cota, use --manual para receber a lista de links.

Exemplos:
    python src/01_baixar_dataset.py                 # baixa tudo
    python src/01_baixar_dataset.py --manual        # so mostra os links
    python src/01_baixar_dataset.py --check         # valida o que ja existe
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------- origens
# IDs do Google Drive publicados no README oficial (wangdongdut/DUT-Anti-UAV).
TRACKING_SOURCES = {
    # Ground truth das 20 sequencias de tracking (videoNN_gt.txt).
    "gt": "16PE3tBhT0lUGZLA8-zIRYvNUvxfhFZJq",
    # Quadros das 20 sequencias de tracking.
    "img": "1dlSPDggg6TRFMcC1jlYIJxxzUQS1mIh9",
}

DETECTION_SOURCES = {
    "train": "1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj",
    "val": "1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n",
    "test": "1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0",
}

BAIDU_TRACKING = {
    "img": ("https://pan.baidu.com/s/1OTExqKgvUnqpENtTDu_gGQ", "oine"),
    "gt": ("https://pan.baidu.com/s/1nkGNERDVgmYIAiwFTdj2xA", "e8mr"),
}

# Contagens esperadas, conferidas contra o dataset usado no artigo.
EXPECTED_TRACKING_VIDEOS = 20
EXPECTED_TRACKING_FRAMES = 24804


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path("datasets/_bruto"),
                        help="Diretorio de destino dos arquivos baixados.")
    parser.add_argument("--what", choices=("all", "tracking", "detection"), default="all")
    parser.add_argument("--manual", action="store_true", help="Apenas imprime os links oficiais.")
    parser.add_argument("--check", action="store_true", help="Apenas valida o que ja foi baixado.")
    parser.add_argument("--testar-links", action="store_true",
                        help="Confere se cada link oficial ainda responde antes de baixar.")
    parser.add_argument("--link", nargs="*", default=None, metavar="NOME=ID_OU_URL",
                        help="Substituir um link, ex.: --link img=1AbC... (util se o Drive mudar).")
    parser.add_argument("--no-extract", action="store_true", help="Nao descompacta os .zip.")
    return parser.parse_args()


def links_alive(sources: dict[str, str]) -> int:
    """Testa cada link oficial. Link do Drive que 'morreu' responde com pagina de erro."""
    import urllib.error
    import urllib.request

    problems = 0
    print(f"{'origem':10s} {'situacao':12s} detalhe")
    print("-" * 78)
    for name, file_id in sources.items():
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        request = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                # O Drive responde a um HEAD com uma pagina de confirmacao, nao
                # com o arquivo: o que importa aqui e o codigo HTTP (link vivo).
                print(f"{name:10s} {'OK':12s} HTTP {response.status} (id {file_id})")
        except urllib.error.HTTPError as error:
            print(f"{name:10s} {'ERRO HTTP':12s} {error.code} ({file_id})")
            problems += 1
        except Exception as error:  # noqa: BLE001
            print(f"{name:10s} {'INACESSIVEL':12s} {type(error).__name__}: {error}")
            problems += 1
    if problems:
        print("\nAlguns links oficiais nao responderam. Causas comuns:")
        print("  - o Google Drive bloqueia downloads muito repetidos (cota excedida):")
        print("    espere algumas horas, ou use o espelho do Baidu, ou baixe pelo navegador;")
        print("  - o link mudou: veja o README oficial em")
        print("    https://github.com/wangdongdut/DUT-Anti-UAV e passe --link nome=ID_NOVO.")
    return problems


def print_links(dest: Path) -> None:
    print("=" * 78)
    print("DUT Anti-UAV - origens oficiais (google Drive dos autores)")
    print("=" * 78)
    print("\nTRACKING (usado no PASSO 4):")
    for name, file_id in TRACKING_SOURCES.items():
        print(f"  {name:4s} https://drive.google.com/open?id={file_id}")
    print("\nDETECTION (usado no PASSO 3):")
    for name, file_id in DETECTION_SOURCES.items():
        print(f"  {name:5s} https://drive.google.com/open?id={file_id}")
    print("\nAlternativa (Baidu, senhas entre parenteses):")
    for name, (url, code) in BAIDU_TRACKING.items():
        print(f"  tracking {name:4s} {url}  ({code})")
    print(f"\nExtraia tudo em: {dest.resolve()}")
    print("Estrutura aceita por este pipeline:")
    print("  <dest>/tracking/gt/videoNN_gt.txt      (ground truth, x y w h por linha)")
    print("  <dest>/tracking/img/*/                 (quadros, .jpg)")
    print("  <dest>/detection/{train,val,test}/{img,xml}/")


def ensure_gdown() -> bool:
    if shutil.which("gdown") is not None:
        return True
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def _baixar_arquivo(gdown, file_id: str, out: Path) -> bool:
    """Baixa UM arquivo do Drive de forma compativel com gdown 4.x, 5.x e 6.x.

    O parametro 'fuzzy' existia no gdown <= 5 e foi REMOVIDO no 6. Passa-lo a
    forca quebra a chamada com TypeError('unexpected keyword argument'). Como o
    codigo nao pode depender da versao instalada na maquina do leitor, aqui a
    chamada e montada a partir da assinatura real do gdown em uso.

    Retorna True se o arquivo foi gravado.
    """
    import inspect

    kwargs = {"id": file_id, "output": str(out), "quiet": False}
    try:
        params = inspect.signature(gdown.download).parameters
    except (TypeError, ValueError):
        params = {}
    if "fuzzy" in params:  # gdown <= 5
        kwargs["fuzzy"] = True
    gdown.download(**kwargs)
    return out.exists()


def _nome_original(gdown, file_id: str) -> str | None:
    """Pergunta ao Drive o nome real do arquivo (sem baixar nada)."""
    try:
        meta = gdown.download(id=file_id, output="/dev/null", quiet=True,
                              skip_download=True)
    except Exception:  # noqa: BLE001
        return None
    path = getattr(meta, "path", None)
    return Path(path).name if path else None


def download(what: str, dest: Path) -> None:
    """Baixa via gdown, tratando cada origem como ARQUIVO unico.

    Os IDs oficiais do README sao todos arquivos .zip (conferido com
    --testar-links). O codigo antigo tentava 'fuzzy=True' e, ao falhar, chamava
    download_folder() -- que em gdown 6.x fica tentando rebaixar a arvore inteira
    em silencio e nao gera zip nenhum. Agora o modo arquivo e o caminho normal.
    """
    import gdown

    grupos = []
    if what in ("all", "tracking"):
        grupos.append(("tracking", dest / "tracking", TRACKING_SOURCES))
    if what in ("all", "detection"):
        grupos.append(("detection", dest / "detection", DETECTION_SOURCES))

    falhas = []
    for rotulo, target, sources in grupos:
        target.mkdir(parents=True, exist_ok=True)
        for name, file_id in sources.items():
            out = target / f"{name}.zip"
            if out.exists() and out.stat().st_size > 0:
                print(f"[{rotulo}/{name}] ja baixado, pulando.")
                continue
            # O Drive pode devolver um QR de confirmacao em vez do arquivo; se o
            # zip ficar minusculo, o download nao valeu.
            original = _nome_original(gdown, file_id)
            if original:
                print(f"[{rotulo}/{name}] arquivo no Drive: {original}")
            print(f"[{rotulo}/{name}] baixando {file_id} ...")
            try:
                ok = _baixar_arquivo(gdown, file_id, out)
            except Exception as error:  # noqa: BLE001
                falhas.append((f"{rotulo}/{name}", f"{type(error).__name__}: {error}"))
                print(f"[{rotulo}/{name}] FALHOU: {type(error).__name__}: {error}")
                continue
            if not ok:
                falhas.append((f"{rotulo}/{name}", "nenhum arquivo gravado"))
                continue
            tamanho = out.stat().st_size / 1e6
            if tamanho < 0.05:
                falhas.append((f"{rotulo}/{name}", f"arquivo suspeito ({tamanho:.3f} MB)"))
                print(f"[{rotulo}/{name}] AVISO: so {tamanho:.3f} MB -- provavel pagina "
                      f"de cota excedida do Drive, nao o dataset.")
                out.unlink(missing_ok=True)
                continue
            print(f"[{rotulo}/{name}] OK ({tamanho:.1f} MB)")

    if falhas:
        print()
        print("=" * 78)
        print("Origens que NAO foram baixadas:")
        for nome, motivo in falhas:
            print(f"  - {nome}: {motivo}")
        print("=" * 78)
        print("O Google Drive limita downloads anonimos (cota diaria). Se o motivo")
        print("acima for de cota, espere algumas horas ou use --manual e baixe pelo")
        print("navegador; os links nao expiram.")


def extract(dest: Path) -> None:
    for archive in sorted(dest.rglob("*.zip")):
        target = archive.with_suffix("")
        if target.exists() and any(target.iterdir()):
            print(f"[extract] {archive.name}: destino ja existe, pulando.")
            continue
        print(f"[extract] {archive.name} -> {target}")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(target)


def check(dest: Path) -> int:
    """Confere a estrutura e as contagens esperadas. Retorna o numero de problemas."""
    problems = 0
    print("=" * 78)
    print(f"Validacao de {dest.resolve()}")
    print("=" * 78)

    # O pacote GT e o pacote de imagens trazem copias do ground truth; conta-se
    # uma vez por sequencia (mesma regra usada pelos passos 2 e 5).
    from uav_common import official_gt_files
    complete = list(official_gt_files(dest).values()) if dest.exists() else []
    frames = 0
    for path in complete:
        frames += sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    if len(complete) == EXPECTED_TRACKING_VIDEOS:
        print(f"  OK   ground truth: {len(complete)} sequencias")
    else:
        print(f"  ERRO ground truth: {len(complete)} sequencias (esperado {EXPECTED_TRACKING_VIDEOS})")
        problems += 1

    if frames == EXPECTED_TRACKING_FRAMES:
        print(f"  OK   quadros no ground truth: {frames}")
    elif frames:
        print(f"  AVISO quadros no ground truth: {frames} (o artigo usa {EXPECTED_TRACKING_FRAMES})")
    else:
        print("  ERRO nenhum ground truth encontrado (tracking/gt).")
        problems += 1

    images = [p for p in dest.rglob("*.jpg")
              if "detection" not in {part.lower() for part in p.parts} and not p.name.startswith("._")]
    print(f"  info quadros de tracking encontrados (jpg): {len(images)}")
    if not images:
        print("  AVISO nenhuma imagem de tracking encontrada (tracking/img).")
    elif len(images) < EXPECTED_TRACKING_FRAMES:
        print(f"  AVISO o artigo usa {EXPECTED_TRACKING_FRAMES} quadros de tracking; "
              "a extracao parece incompleta (o PASSO 2 tambem aceita os .mp4).")

    xmls = list(dest.rglob("*.xml"))
    print(f"  info anotacoes de deteccao encontradas (xml): {len(xmls)}")
    if xmls and len(xmls) < 10000:
        print(f"  AVISO o conjunto de deteccao tem 10.000 imagens; encontradas {len(xmls)}.")

    return problems


def main() -> int:
    args = parse_args()
    dest = args.dest.resolve()

    if args.link:
        for item in args.link:
            if "=" not in item:
                raise SystemExit(f"--link espera Nome=ID_OU_URL; recebi '{item}'")
            name, value = item.split("=", 1)
            # Aceita tanto o ID puro quanto a URL completa do Drive.
            file_id = value.split("id=")[-1] if "drive.google.com" in value else value
            if name in TRACKING_SOURCES:
                TRACKING_SOURCES[name] = file_id
            elif name in DETECTION_SOURCES:
                DETECTION_SOURCES[name] = file_id
            else:
                raise SystemExit(f"origem desconhecida '{name}' (use: "
                                 f"{', '.join(list(TRACKING_SOURCES) + list(DETECTION_SOURCES))})")
            print(f"link de '{name}' substituido por {file_id}")

    if args.manual:
        print_links(dest)
        return 0

    if args.check:
        return 1 if check(dest) else 0

    if args.testar_links:
        problems = links_alive({**TRACKING_SOURCES, **DETECTION_SOURCES})
        return 1 if problems else 0

    if not ensure_gdown():
        print("O utilitario 'gdown' e necessario para baixar do Google Drive.\n")
        print("  pip install gdown\n")
        print("Depois repita este comando. Se o Drive estiver com cota excedida,")
        print("rode com --manual e baixe pelo navegador (os links nao expiram).")
        print_links(dest)
        return 2

    download(args.what, dest)
    if not args.no_extract:
        extract(dest)
    return 1 if check(dest) else 0


if __name__ == "__main__":
    sys.exit(main())

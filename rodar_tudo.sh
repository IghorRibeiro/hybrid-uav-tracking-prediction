#!/usr/bin/env bash
#
# Roda o pipeline do artigo de ponta a ponta: dataset -> rastreadores -> tabelas -> figuras.
#
#   bash rodar_tudo.sh                       # pipeline completo com os pesos do artigo
#   bash rodar_tudo.sh --video video03       # teste rapido em uma sequencia
#   bash rodar_tudo.sh --pular-download      # datasets/_bruto ja esta pronto
#   bash rodar_tudo.sh --treinar             # refaz tambem o ajuste fino do YOLO26n (PASSO 3)
#   bash rodar_tudo.sh --treinar --com-m     # ... e o YOLO26m (linha comparativa da Tabela 1)
#   bash rodar_tudo.sh --com-deepsort        # inclui o DeepSORT (lento: ~4 FPS no Apple M2)
#   bash rodar_tudo.sh --tabela1             # roda o val() dos detectores (Tabela 1)
#   bash rodar_tudo.sh --device mps          # forca o dispositivo (padrao: auto)
#
# Variaveis de ambiente: PYTHON, RAW (padrao datasets/_bruto), EPOCHS (50), IMGSZ (640).
#
# Cada passo confere o anterior: se algo faltar, o script para ali em vez de
# produzir numeros silenciosamente errados.

set -uo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
RAW="${RAW:-datasets/_bruto}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
DEVICE="auto"
VIDEO=""
PULAR_DOWNLOAD=0
TREINAR=0
COM_M=0
COM_DEEPSORT=0
TABELA1=0

while [ $# -gt 0 ]; do
  case "$1" in
    --raw)             RAW="$2"; shift 2 ;;
    --video|-v)        VIDEO="$2"; shift 2 ;;
    --device|-d)       DEVICE="$2"; shift 2 ;;
    --python)          PYTHON="$2"; shift 2 ;;
    --pular-download)  PULAR_DOWNLOAD=1; shift ;;
    --treinar)         TREINAR=1; shift ;;
    --com-m)           COM_M=1; shift ;;
    --com-deepsort)    COM_DEEPSORT=1; shift ;;
    --tabela1)         TABELA1=1; shift ;;
    -h|--help)         sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "opcao desconhecida: $1"; exit 2 ;;
  esac
done

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
  echo "usando o interpretador da .venv local"
fi

# bash 3.2 (macOS) + 'set -u': expandir array vazio e erro; o idioma
# ${ARR[@]+"${ARR[@]}"} expande para nada quando o array esta vazio.
VIDEO_ARG=()
[ -n "$VIDEO" ] && VIDEO_ARG=(--video "$VIDEO")

MODEL="pesos_treinados/yolo26n_dut_best.pt"   # pesos do artigo (versionados)
PRED="results/predictions"

banner() {
  echo
  echo "=============================================================================="
  echo "  $1"
  echo "=============================================================================="
}
falhou() { echo; echo ">>> O passo anterior terminou com erro. Corrija antes de continuar."; exit 1; }
rastrear() {  # rastrear <args do 04_rastreadores.py>
  "$PYTHON" src/04_rastreadores.py --device "$DEVICE" --model "$MODEL" "$@" \
    ${VIDEO_ARG[@]+"${VIDEO_ARG[@]}"}
}

# ---------------------------------------------------------------- ambiente
banner "VERIFICACAO DO AMBIENTE"
"$PYTHON" check_ambiente.py || true

# ------------------------------------------------------------------ PASSO 1
O_QUE="tracking"
[ "$TREINAR" -eq 1 ] && O_QUE="all"   # o treino precisa do conjunto de deteccao
if [ "$PULAR_DOWNLOAD" -eq 0 ]; then
  banner "PASSO 1/6 - Baixando o DUT Anti-UAV da origem oficial ($O_QUE)"
  "$PYTHON" src/01_baixar_dataset.py --dest "$RAW" --what "$O_QUE" || falhou
else
  banner "PASSO 1/6 - Download pulado; conferindo $RAW"
  "$PYTHON" src/01_baixar_dataset.py --dest "$RAW" --check || {
    [ -n "$VIDEO" ] && echo "  (conjunto incompleto, mas --video foi pedido: seguindo)" || falhou; }
fi

# ------------------------------------------------------------------ PASSO 2
banner "PASSO 2/6 - Convertendo para o formato YOLO"
"$PYTHON" src/02_converter_dataset.py --raw "$RAW" --only tracking || falhou
if [ ! -f "datasets/anti_uav_tracking_test_final/resolutions.json" ]; then
  echo ">>> ERRO: resolutions.json nao foi gerado; confira o material em $RAW."
  exit 1
fi
if [ "$TREINAR" -eq 1 ]; then
  "$PYTHON" src/02_converter_dataset.py --raw "$RAW" --only detection || falhou
fi

# ------------------------------------------------------------------ PASSO 3
if [ "$TREINAR" -eq 1 ]; then
  banner "PASSO 3/6 - Ajuste fino do YOLO26n no DUT Anti-UAV (deteccao)"
  "$PYTHON" src/03_treinar_detector.py --model yolo26n.pt --name yolo26_n \
    --epochs "$EPOCHS" --imgsz "$IMGSZ" || falhou
  MODEL="pesos_treinados/treinados_localmente/yolo26n_dut_best.pt"
  if [ "$COM_M" -eq 1 ]; then
    "$PYTHON" src/03_treinar_detector.py --model yolo26m.pt --name yolo26_m \
      --epochs "$EPOCHS" --imgsz "$IMGSZ" || falhou
  fi
else
  banner "PASSO 3/6 - Usando os pesos do artigo ($MODEL)"
fi
[ -f "$MODEL" ] || { echo ">>> ERRO: pesos nao encontrados em $MODEL"; exit 1; }

# ------------------------------------------------------------------ PASSO 4
banner "PASSO 4/6 - Metodo proposto (YOLO26n + CSRT + Kalman, N_red = 15)"
rastrear --tracker hibrido --local-tracker csrt --period 15 || falhou

banner "PASSO 4/6 - Rastreadores locais alternativos (Tabela 2)"
for local in kcf mosse medianflow; do
  rastrear --tracker hibrido --local-tracker "$local" --period 15 || \
    echo "  (aviso: $local falhou; confira se o opencv-contrib-python esta instalado)"
done

banner "PASSO 4/6 - Baselines de deteccao + associacao (Tabela 3)"
for tracker in botsort bytetrack; do
  rastrear --tracker "$tracker" || echo "  (aviso: $tracker falhou)"
done
if [ "$COM_DEEPSORT" -eq 1 ]; then
  rastrear --tracker deepsort || echo "  (aviso: deepsort falhou; veja requirements.txt)"
fi

# ------------------------------------------------------------------ PASSO 5
banner "PASSO 5/6 - Tabela 2 (rastreadores locais dentro do pipeline)"
"$PYTHON" src/05_avaliar.py --gt-dir "$RAW" --output-dir results/tabela2 --referencia CSRT \
  --metodos "CSRT=$PRED/hibrido" "KCF=$PRED/hibrido_kcf" \
            "MOSSE=$PRED/hibrido_mosse" "MedianFlow=$PRED/hibrido_medianflow" || falhou

banner "PASSO 5/6 - Tabela 3 (comparacao com rastreadores da literatura)"
METODOS_T3=("BoT-SORT=$PRED/botsort")
[ -d "$PRED/deepsort" ] && METODOS_T3+=("DeepSORT=$PRED/deepsort")
METODOS_T3+=("ByteTrack=$PRED/bytetrack")
[ -d "$PRED/ostrack" ] && METODOS_T3+=("OSTrack=$PRED/ostrack")   # src/extras/
[ -d "$PRED/odtrack" ] && METODOS_T3+=("ODTrack=$PRED/odtrack")   # src/extras/
METODOS_T3+=("Hibrido=$PRED/hibrido")
"$PYTHON" src/05_avaliar.py --gt-dir "$RAW" --output-dir results/tabela3 --referencia BoT-SORT \
  --metodos "${METODOS_T3[@]}" || falhou

banner "PASSO 5/6 - Figura 3 (Kalman persistente x Kalman Damped, gamma = 0,95)"
"$PYTHON" src/05_avaliar.py --gt-dir "$RAW" --output-dir results/figura3 \
  --metodos "Hibrido=$PRED/hibrido" --ablar-damping 1.0 0.95 || falhou

if [ "$TABELA1" -eq 1 ]; then
  banner "PASSO 5/6 - Tabela 1 (detectores nas 24.804 imagens de tracking)"
  PESOS_M="pesos_treinados/treinados_localmente/yolo26m_dut_best.pt"
  "$PYTHON" src/05_avaliar.py --tabela1 --output-dir results/tabela1 --device "$DEVICE" \
    --pesos-yolo26n "$MODEL" --pesos-yolo26m "$PESOS_M" || falhou
fi

# ------------------------------------------------------------------ PASSO 6
banner "PASSO 6/6 - Figuras"
"$PYTHON" src/06_figuras.py --tabelas results/tabela3 --ablacao results/figura3 \
  --figuras results/figures --gt-dir "$RAW" \
  --metodos "Hibrido=$PRED/hibrido" "BoT-SORT=$PRED/botsort" "ByteTrack=$PRED/bytetrack" || \
  echo "  (aviso: figuras nao foram geradas; as tabelas do PASSO 5 estao prontas)"

banner "FIM"
echo "Tabelas : results/tabela2/  results/tabela3/  results/figura3/  (e results/tabela1/)"
echo "Figuras : results/figures/"
echo "Predicoes: $PRED/"
echo
echo "Os numeros sao os do SEU ambiente. Veja docs/DIFERENCAS_ENTRE_AMBIENTES.md para"
echo "entender o que muda entre maquinas e o que deve se manter (ordem de grandeza)."

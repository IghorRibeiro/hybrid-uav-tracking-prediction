# Do artigo ao comando

Para cada tabela e figura do artigo, o comando que a reproduz com os seus próprios números. Os comandos assumem que os passos 1 e 2 já foram executados (dataset em `datasets/_bruto` e quadros convertidos em `datasets/anti_uav_tracking_test_final`) e que você está na raiz do repositório.

O `rodar_tudo.sh` executa exatamente esta sequência.

---

## Pré-requisito: os dados

```bash
python src/01_baixar_dataset.py --what tracking      # 20 sequências + ground truth oficial
python src/02_converter_dataset.py --only tracking   # quadros padronizados + rótulos YOLO
```

Conferência esperada no fim do passo 2: **20 sequências, 24.804 quadros, 2.586 sem alvo**. Quinze sequências são 1920×1080 e cinco são 1280×720 (`video02`, `video03`, `video05`, `video07`, `video08`); a resolução de cada uma fica em `resolutions.json`.

---

## Tabela 1: detectores YOLO26n e YOLO26m

```bash
# opcional: refazer o ajuste fino (o repositório já traz o YOLO26n do artigo)
python src/02_converter_dataset.py --only detection
python src/03_treinar_detector.py --model yolo26n.pt --name yolo26_n
python src/03_treinar_detector.py --model yolo26m.pt --name yolo26_m

python src/05_avaliar.py --tabela1 --output-dir results/tabela1 \
  --pesos-yolo26n pesos_treinados/yolo26n_dut_best.pt \
  --pesos-yolo26m pesos_treinados/treinados_localmente/yolo26m_dut_best.pt
```

* **Saída:** `results/tabela1/tabela1_detectores.csv` (precisão, recall, mAP@0.5, mAP@0.5:0.95).
* **Protocolo do artigo:** ajuste fino no conjunto *detection* do DUT Anti-UAV (50 épocas, imgsz 640, batch 8, `amp=False`); avaliação com o `val()` do Ultralytics sobre as **24.804 imagens das 20 sequências de tracking**.
* O YOLO26m entra só nesta tabela. Sem os pesos dele, a tabela sai só com a linha do YOLO26n.

---

## Tabela 2: rastreadores locais dentro do pipeline híbrido

```bash
for t in csrt kcf mosse medianflow; do
  python src/04_rastreadores.py --tracker hibrido --local-tracker $t --period 15
done
python src/05_avaliar.py --gt-dir datasets/_bruto --output-dir results/tabela2 --referencia CSRT \
  --metodos CSRT=results/predictions/hibrido \
            KCF=results/predictions/hibrido_kcf \
            MOSSE=results/predictions/hibrido_mosse \
            MedianFlow=results/predictions/hibrido_medianflow
```

* **Saída:** `results/tabela2/comparacao.csv` (SA, SR@0.5, H30 média ± desvio, FPS), além de `por_sequencia.csv`, `resumo.csv` e `testes_pareados.csv`.
* O CSRT grava em `results/predictions/hibrido/` porque é a configuração adotada; os demais rastreadores locais gravam em pastas próprias.
* **FPS:** é a métrica menos transportável. Compare apenas entre métodos rodados na mesma máquina e na mesma sessão.

---

## Tabela 3: comparação com rastreadores da literatura

```bash
python src/04_rastreadores.py --tracker hibrido --local-tracker csrt --period 15
python src/04_rastreadores.py --tracker botsort
python src/04_rastreadores.py --tracker bytetrack
python src/04_rastreadores.py --tracker deepsort          # lento; requer deep-sort-realtime

# opcional: baselines Transformer (repositórios e pesos externos)
python src/extras/rodar_ostrack_odtrack.py --tracker ostrack --repo externos/OSTrack \
  --param vitb_384_mae_ce_32x4_ep300 --gt-dir datasets/_bruto
python src/extras/rodar_ostrack_odtrack.py --tracker odtrack --repo externos/ODTrack \
  --param baseline --gt-dir datasets/_bruto

python src/05_avaliar.py --gt-dir datasets/_bruto --output-dir results/tabela3 --referencia BoT-SORT \
  --metodos BoT-SORT=results/predictions/botsort \
            DeepSORT=results/predictions/deepsort \
            ByteTrack=results/predictions/bytetrack \
            OSTrack=results/predictions/ostrack \
            ODTrack=results/predictions/odtrack \
            Hibrido=results/predictions/hibrido
```

* **Saída:** `results/tabela3/comparacao.csv` com SA, SR@0.5, IoU médio, AUC, H30 (média ± desvio e IC95%), ganho percentual de H30 sobre o BoT-SORT e FPS.
* BoT-SORT, ByteTrack e DeepSORT usam o mesmo YOLO26n do híbrido e a configuração padrão de cada rastreador. OSTrack e ODTrack são inicializados com a caixa do primeiro quadro do ground truth.
* Todas as trajetórias passam pelo mesmo Kalman persistente (`--damping 1.0`, padrão), de modo que a diferença no H30 vem do rastreamento e não do preditor.
* Métodos ausentes são ignorados com aviso: a tabela sai com os métodos que você rodou.
* O texto do artigo cita os intervalos de confiança de 95% do H30 (Híbrido [94,3; 185,5] px, BoT-SORT [84,1; 368,3] px). Eles estão na coluna `H30_IC95_px` e usam t de Student com n = 20 sequências.

---

## Testes estatísticos

Gerados junto com qualquer tabela em `testes_pareados.csv`: t pareado e Wilcoxon por sequência contra a referência, com IC95% da diferença por bootstrap. A unidade amostral é a **sequência** (n = 20), porque quadros de um mesmo vídeo são fortemente correlacionados.

---

## Figura 3: Kalman persistente × Kalman Damped

```bash
python src/05_avaliar.py --gt-dir datasets/_bruto --output-dir results/figura3 \
  --metodos Hibrido=results/predictions/hibrido --ablar-damping 1.0 0.95
python src/06_figuras.py --tabelas results/tabela3 --ablacao results/figura3 --figuras results/figures
```

* **Saída:** `results/figura3/ablacao_damping.csv` (H5 a H30, média e desvio entre sequências, redução percentual) e `results/figures/fig3_kalman_damped.png|pdf`.
* A ablação relê **as mesmas predições** e recalcula apenas o preditor: a única variável é o amortecimento. Passe outros fatores (`--ablar-damping 1.0 0.95 0.90 0.85`) para ver a tendência.
* O amortecimento γ é aplicado à velocidade após cada correção e **a cada passo da predição**. Essa é a formulação que reproduz a redução do artigo (ver `docs/VALIDACAO.md`); aplicar γ só uma vez por quadro dá uma redução bem menor.

---

## Figuras 1 e 2

São esquemas e exemplos qualitativos do pipeline e não carregam números. A Figura 1 está em [`docs/pipeline.png`](pipeline.png).

---

## Afirmações do texto

| No artigo | Onde conferir |
|---|---|
| 20 sequências, 24.804 quadros | saída do `02_converter_dataset.py --only tracking` |
| Redetecção a cada 15 quadros | `--period 15` (padrão) no `04_rastreadores.py` |
| Confiança 0,5 para inicializar e 0,7 para reancorar | `--init-conf` e `--switch-conf` |
| Kalman CV, estado `[x, y, ẋ, ẏ]`, Δt = 1 | `init_kalman()` em `src/uav_common.py` |
| γ = 0,95 no Kalman Damped | `--ablar-damping 1.0 0.95` no `05_avaliar.py` |
| Horizontes H5, H10, H15 e H30 | `HORIZONS` em `src/uav_common.py` |
| Ganho de H30 sobre o BoT-SORT | coluna `Ganho_H30_%_vs_BoT-SORT` da Tabela 3 |
| FPS no Apple M2 (MPS) | coluna `FPS`, medida na sua máquina |

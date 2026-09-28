## O que deve se manter

* **Ordem de grandeza** de SA, SR@0.5, IoU, AUC e erro de predição (H5 a H30).
* **Comportamento relativo** entre métodos avaliados na mesma execução: por exemplo, o híbrido com CSRT próximo dos baselines em SR@0.5, o Kalman Damped reduzindo o erro em H30 em relação ao persistente e o DeepSORT bem mais lento que os demais.
* **O protocolo:** mesmas 20 sequências, mesmos 24.804 quadros, mesmo ground truth, mesmos limiares, mesmas definições de métrica e o mesmo pareamento por sequência nos testes estatísticos.

## O que muda, em ordem de impacto

| Fator | Efeito típico | Controle |
|---|---|---|
| Pesos do detector | outro `best.pt` muda quais detecções passam do limiar | use os pesos do artigo (`pesos_treinados/yolo26n_dut_best.pt`); se treinar, reporte |
| Versão do Ultralytics | pré e pós-processamento e configuração padrão do BoT-SORT/ByteTrack mudam entre versões | anote a versão (o `check_ambiente.py` imprime) |
| Dispositivo (CUDA, MPS, CPU) | kernels diferentes fazem detecções limítrofes mudarem de lado | informe o `--device` usado |
| Versão do OpenCV | CSRT, KCF, MOSSE e MedianFlow mudaram entre versões | use `opencv-contrib-python`; o artigo usou 4.13 |
| Quadros extraídos de vídeo | recompressão gera detecções levemente diferentes dos JPEGs originais | prefira o pacote oficial de imagens |
| FPS | depende do processador, da carga da máquina e do dispositivo | compare só entre métodos da mesma execução |

Um rastreador híbrido é sensível a pequenas diferenças: se o detector deixa de reancorar num quadro, o CSRT segue por mais 15 quadros com o erro acumulado, e isso aparece no SA e no H30 daquela sequência. Por isso a variação entre execuções é maior no rastreamento do que na detecção (o mAP da Tabela 1 é bem estável).

Exemplo real deste projeto: em duas rodadas do pipeline híbrido feitas pelo autor, o MOSSE deu SR@0.5 de 0,811 e 0,823 e o MedianFlow, de 0,736 e 0,787 (detalhes em `docs/VALIDACAO.md`). É esse o tamanho de variação que se deve esperar entre execuções.

## O que já está fixado no código

* Treino: `epochs=50`, `imgsz=640`, `batch=8`, `patience=10`, `optimizer="auto"`, `amp=False`, `seed=0`, `deterministic=True`.
* Híbrido: `N_red = 15`, confiança 0,5 para inicializar e 0,7 para reancorar, caixa de maior confiança por quadro.
* Kalman CV: `processNoiseCov = 0,03·I`, `measurementNoiseCov = 0,1·I`, `errorCovPost = I` na inicialização; amortecimento aplicado a cada passo da predição.
* Avaliação: SR@0.5 com IoU ≥ 0,5, 21 limiares de IoU para a AUC, horizontes H5, H10, H15 e H30, ground truth oficial com a resolução real de cada sequência.

## Como relatar os seus números

1. Onde rodou (Colab, local, servidor) e o dispositivo (`cuda`, `mps`, `cpu`).
2. As versões de `torch`, `ultralytics` e `opencv-contrib-python` (`python check_ambiente.py`).
3. Os pesos usados (os do artigo ou treinados por você).
4. Os comandos exatos, inclusive `--period` e `--damping`.

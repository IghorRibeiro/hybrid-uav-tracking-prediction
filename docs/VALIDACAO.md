# Validação do pipeline

Este documento registra como o código deste repositório foi conferido contra o artigo. Os resultados do autor **não** fazem parte do repositório; os números abaixo servem só para mostrar que o código mede o mesmo que foi medido no artigo.

## 1. O avaliador reproduz as tabelas a partir das predições originais

O `src/05_avaliar.py` foi aplicado aos arquivos de predição gerados pelo autor para o artigo (um `videoNN.txt` por método e sequência), contra o ground truth oficial do DUT Anti-UAV. Se o avaliador medisse algo diferente do artigo, as tabelas não bateriam.

**Tabela 3** (Kalman persistente, 20 sequências, H30 em px):

| Método | SA | SR@0.5 | IoU | AUC | H30 | Ganho | FPS | Artigo |
|---|---|---|---|---|---|---|---|---|
| BoT-SORT | 0,828 | 0,832 | 0,659 | 0,662 | 226,2 ± 303,6 | – | 42,7 | idêntico |
| DeepSORT | 0,830 | 0,834 | 0,662 | 0,664 | 261,9 ± 446,0 | −15,8 | 4,5 | idêntico |
| ByteTrack | 0,833 | 0,836 | 0,660 | 0,663 | 270,1 ± 462,9 | −19,4 | 58,7 | H30 269,1 e ganho −19,0 no artigo |
| OSTrack | 0,803 | 0,831 | 0,668 | 0,672 | 204,4 ± 206,3 | +9,6 | 13,7 | H30 204,3 e ganho +9,7 no artigo |
| ODTrack | 0,819 | 0,860 | 0,707 | 0,709 | 167,4 ± 160,8 | +26,0 | 7,3 | idêntico |
| Híbrido | 0,786 | 0,811 | 0,646 | 0,648 | 139,8 ± 97,6 | +38,2 | * | SA 0,789, IoU 0,645, AUC 0,647 e H30 139,9 ± 97,4 no artigo |

\* O arquivo de predições do híbrido usado no artigo não registrou o tempo por quadro, então o FPS não pode ser recalculado a partir dele. O pipeline atual grava o tempo de cada quadro, e o FPS sai do próprio arquivo.

Os IC95% do H30 também coincidem com os citados no texto: BoT-SORT [84,1; 368,3] px e Híbrido [94,1; 185,5] px (artigo: [94,3; 185,5]).

**Tabela 2** (rastreadores locais dentro do pipeline):

| Rastreador | SA | SR@0.5 | H30 | Artigo |
|---|---|---|---|---|
| CSRT | 0,786 | 0,811 | 139,8 ± 97,6 | 0,7858 / 0,8108 / 139,92 ± 97,44 |
| KCF | 0,740 | 0,741 | 179,5 ± 189,4 | 0,7394 / 0,7409 / 179,47 ± 189,35 |
| MOSSE | 0,819 | 0,823 | 197,9 ± 248,1 | 0,8050 / 0,8108 / 191,80 ± 228,68 |
| MedianFlow | 0,773 | 0,787 | 162,2 ± 126,4 | 0,7226 / 0,7356 / 168,99 ± 121,08 |

CSRT e KCF batem com o artigo. Os arquivos de MOSSE e MedianFlow disponíveis são de uma rodada posterior à da tabela; a diferença (1 a 5 centésimos no SR@0.5) é um bom exemplo da variação que se deve esperar entre execuções do mesmo método.

## 2. O Kalman Damped reproduz a redução da Figura 3

Com as mesmas predições do híbrido, variando só o preditor:

| | H5 | H10 | H15 | H30 | Redução em H30 |
|---|---|---|---|---|---|
| Persistente (γ = 1,0) | 36,7 | 53,9 | 73,7 | 139,8 | – |
| Damped (γ = 0,95), γ a cada passo da predição | 36,0 | 51,2 | 67,0 | 111,1 | 20,5% |
| Damped (γ = 0,95), γ só após a correção | 36,1 | 52,2 | 70,6 | 131,4 | 6,0% |
| Artigo | 30,2 | 46,4 | 63,1 | 109,1 | 22,0% |

A formulação com γ aplicado a cada passo da predição é a que reproduz a redução do artigo, e é a implementada em `src/uav_common.py`. Os valores do artigo vieram de outra rodada do híbrido, o que explica a diferença em H5.

## 3. O rastreador reproduz o comportamento do artigo

O `src/04_rastreadores.py` implementa a mesma lógica do rastreador usado no artigo (inicialização com confiança 0,5, reancoragem a cada 15 quadros com confiança 0,7, reinicialização do CSRT quando ele falha). Rodado em CPU (Linux, Ultralytics 8.4, OpenCV 4.13) sobre `video02` e `video03`, com os pesos do artigo:

| | SR@0.5 médio | H30 médio |
|---|---|---|
| Este repositório (CPU) | 0,975 | 40,1 px |
| Predições do artigo (Apple M2) | 0,985 | 39,4 px |

BoT-SORT, ByteTrack e DeepSORT também foram executados em `video03` e ficaram na mesma faixa do artigo.

## 4. Teste de ponta a ponta

O `rodar_tudo.sh --pular-download --video video03` foi executado sobre uma cópia da estrutura oficial dos pacotes do DUT Anti-UAV (`Anti-UAV-Tracking-V0GT/` e `Anti-UAV-Tracking-V0/videoNN/00001.jpg`): conversão, os quatro rastreadores locais, BoT-SORT e ByteTrack, Tabelas 2 e 3, ablação do damping e as cinco figuras, terminando sem erro.

## 5. Notas para quem compara com o texto do artigo

* **SA.** O código usa a definição com limiar: quadros com alvo e IoU ≥ 0,5, mais os quadros sem alvo corretamente declarados sem alvo, divididos pelo total. É a definição das Tabelas 2 e 3. A média simples do IoU corresponde à coluna **IoU médio**.
* **SA do híbrido na Tabela 3.** A média entre as 20 sequências é 0,786 (o mesmo valor da linha CSRT da Tabela 2, que é a configuração adotada); a tabela do artigo traz 0,789.
* **FPS.** Só é comparável entre métodos da mesma execução e na mesma máquina.

## Testes automatizados

`tests/test_uav_common.py` confere as métricas e o Kalman com dados sintéticos (não precisa do dataset):

```bash
python -m pytest tests/       # ou: python tests/test_uav_common.py
```

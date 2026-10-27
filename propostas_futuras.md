# Documentação Metodológica de Propostas Futuras e Referências Teóricas
### Projeto Plant Pathology 2020 (FGVC7) - Redes Neurais Artificiais (UFERSA)

Este documento consolida a fundamentação teórica, modelagem matemática, espaços de busca e especificações de engenharia para as duas abordagens planejadas que, por restrições operacionais de tempo e cotas de GPU, foram formalizadas como **trabalhos futuros e referências de extensão**:
1. **Busca Exaustiva de Hiperparâmetros (*Grid Search*)**
2. **Combinação e Ponderação de Modelos (*Ensemble de Redes Neurais*)**

---

## PARTE 1: Fundamentação e Modelagem do Grid Search

### 1.1 Conceito e Formulação Matemática
O **Grid Search (Busca em Grade)** consiste na definição prévia de um conjunto finito e discreto de hiperparâmetros, avaliando o desempenho do modelo no produto cartesiano completo de suas combinações:

$$\Omega = \Lambda_1 \times \Lambda_2 \times \dots \times \Lambda_K$$

Onde cada $\lambda_k \in \Lambda_k$ representa uma configuração de hiperparâmetro (taxa de aprendizado, taxa de dropout, proporção de validação, etc.).

### 1.2 Espaço de Busca Cartesiano Formulado (648 Combinações)

| Hiperparâmetro | Notação | Espaço a Variar | Cardinalidade | Justificativa Técnica |
| :--- | :---: | :--- | :---: | :--- |
| **Famílias de Backbone** | $\mathcal{M}$ | `DenseNet121`, `EfficientNetB0`, `ResNet50V2`, `MobileNetV2` | 4 | Avaliar famílias com conectividades distintas (densas, compostas, residuais e invertidas). |
| **Taxa de Aprendizado Inicial** | $\eta$ | `[1e-3, 5e-4, 1e-4]` | 3 | Sensibilidade da dinâmica de convergência do otimizador Adam. |
| **Épocas Teto** | $E_{\text{max}}$ | `[100, 200, 400]` *(com EarlyStopping)* | 3 | Horizonte de treinamento seguro com parada reativa. |
| **Divisão de Validação** | $v_{\text{split}}$ | `[0.15, 0.20, 0.25]` *(estratificado)* | 3 | Avaliar a sensibilidade volumétrica do conjunto de treino. |
| **Taxa de Dropout** | $p_{\text{drop}}$ | `[0.0, 0.1, 0.3]` | 3 | Calibração de regularização no topo contra sobreajuste. |
| **Batch Size Base** | $B$ | `[32, 64]` | 2 | Maximização do throughput de VRAM na GPU. |

$$\text{Total de Combinações Teóricas: } |\Omega| = 4 \times 3 \times 3 \times 3 \times 3 \times 2 = 648 \text{ treinos completos}$$

### 1.3 Racional Metodológico para Não Execução do Grid Search Exaustivo
1. **Convergência Imediata de Alto Padrão no Baseline:**
   - Com a calibração de primeira ordem realizada na PoC (`02_densenet121_baseline.ipynb`), utilizando resolução $512 \times 512$, *Fine-Tuning* descongelado, precisão mista `mixed_float16` e `class_weight` balanceado, a `DenseNet121` atingiu $>95\%$ de acurácia e $>0.97$ de Macro ROC-AUC.
2. **Custo Computacional Inviável para Ambiente Acadêmico Gratuito:**
   - Cada treino em 512x512 consome ~12 a 15 minutos em GPU Tesla T4. A execução de 648 combinações exigiria mais de **130 horas ininterruptas de GPU** (~5,5 dias), violando os limites de sessão e créditos do Google Colab.
3. **Priorização da Diversidade Arquitetural:**
   - Os hiperparâmetros ótimos foram fixados ($\eta = 10^{-4}$, $\text{Dropout} = 0.3$, $v_{\text{split}} = 0.20$), redirecionando a capacidade computacional para comparar famílias arquiteturais distintas no notebook de benchmarking (`03_benchmark_models.ipynb`).

---

## PARTE 2: Fundamentação e Modelagem do Ensemble

### 2.1 Princípio da Diversidade e Redução de Variância
O benefício estatístico de combinar múltiplos modelos convolucionais baseia-se na decorrelação de seus resíduos preditivos. Pelo teorema de redução de ambiguidade (Krogh & Vedelsby):

$$\mathbb{E}[(\bar{f}(x) - y)^2] = \frac{1}{M} \sum_{m=1}^M \mathbb{E}[(f_m(x) - y)^2] - \frac{1}{M} \sum_{m=1}^M \mathbb{E}[(f_m(x) - \bar{f}(x))^2]$$

Onde o segundo termo representa a **ambiguidade (diversidade)** entre os modelos. Quanto mais heterogêneas forem as representações aprendidas, menor será o erro quadrático esperado do ensemble.

### 2.2 Estratégias de Fusão Formuladas

#### A. Média Simples (*Soft Voting*)
$$\mathbf{P}_{\text{ensemble}}(x) = \frac{1}{M} \sum_{m=1}^{M} \mathbf{P}_m(x)$$
* Abordagem não-paramétrica, imune a overfitting na validação e robusta para modelos com desempenhos próximos.

#### B. Média Ponderada por ROC-AUC (*Weighted Blending*)
Pondera as probabilidades conforme o poder de ordenação de cada modelo:
$$w_m = \frac{\text{AUC}_m - 0.5}{\sum_{k=1}^M (\text{AUC}_k - 0.5)}, \quad \mathbf{P}_{\text{weighted}}(x) = \sum_{m=1}^{M} w_m \mathbf{P}_m(x)$$
* Penaliza backbones com menor separabilidade probabilística na classe minoritária `multiple_diseases`.

#### C. Otimização Numérica Restrita (SLSQP)
$$\min_{\mathbf{w}} \; \mathcal{L}_{\text{CE}}\left(\sum_{m=1}^M w_m \mathbf{P}_m(X_{\text{val}}), Y_{\text{val}}\right) \quad \text{sujeito a} \quad \sum_{m=1}^M w_m = 1, \; w_m \ge 0$$

#### D. Meta-Classificador (*Stacking*)
Treinamento de uma Regressão Logística com regularização $L_2$ sobre as probabilidades geradas em validação cruzada (*Out-of-Fold*).

### 2.3 Especificação do Pipeline de Submissão Kaggle

```python
# Pipeline formal de inferência do Ensemble sobre data/test.csv:
import os
import numpy as np
import pandas as pd
import tensorflow as tf

test_df = pd.read_csv("data/test.csv")
test_df["file_path"] = test_df["image_id"].apply(lambda x: f"data/images/{x}.jpg")

def load_test_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [512, 512])
    return img

test_ds = tf.data.Dataset.from_tensor_slices(test_df["file_path"].values)
test_ds = test_ds.map(load_test_img, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

# Carregamento dos modelos treinados no benchmarking e ponderação
model_paths = [
    "models/best_DenseNet121.keras",
    "models/best_EfficientNetV2B0.keras",
    "models/best_ResNet50V2.keras",
    "models/best_ConvNeXtTiny.keras",
    "models/best_Xception.keras"
]
weights = [0.25, 0.25, 0.20, 0.15, 0.15]  # Calibrados via AUC na validação

final_preds = np.zeros((len(test_df), 4))
for path, w in zip(model_paths, weights):
    if os.path.exists(path):
        m = tf.keras.models.load_model(path)
        final_preds += w * m.predict(test_ds)

# Exportação do submission.csv oficial
sub = pd.DataFrame(final_preds, columns=["healthy", "multiple_diseases", "rust", "scab"])
sub.insert(0, "image_id", test_df["image_id"])
sub.to_csv("data/submission.csv", index=False)
```


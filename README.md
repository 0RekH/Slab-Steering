# T-Lab Interpretability — Improving Activation Steering with Learned Denoisers

Этот репозиторий содержит решение исследовательского задания по interpretability, посвящённого улучшению activation steering с помощью обучаемых denoiser / repair-модулей.

Основная идея работы заключается в следующем: стандартный activation steering позволяет усиливать выбранный семантический концепт в скрытых представлениях языковой модели, однако при слишком сильной интервенции качество генерации резко ухудшается.

В этой работе исследуется, можно ли восстановить скрытые состояния после steering так, чтобы сохранить полезный семантический эффект и одновременно уменьшить деградацию качества текста.

В качестве базовой модели используется GPT-2 Small, а steering-направление извлекается из Sparse Autoencoder.

---

## Основная постановка

Стандартный activation steering модифицирует скрытое состояние модели следующим образом:

```math
\widetilde h = h + \alpha v
```

где:

- $h$ — скрытое представление модели;
- $v$ — steering direction;
- $\alpha$ — сила интервенции.

При небольших значениях $\alpha$ можно усилить нужный концепт практически без потери качества.

Однако при увеличении $\alpha$ активации начинают уходить из естественного распределения hidden states модели, что приводит к росту perplexity, ухудшению связности текста и, в предельном случае, к полному collapse генерации.

Цель работы — улучшить компромисс между двумя величинами:

- выраженностью целевого концепта;
- качеством генерации.

Иными словами, хочется получить одновременно:

```math
\text{Concept score} \uparrow
```

и

```math
\text{PPL} \downarrow
```

---

## Используемая модель

В экспериментах используется:

- GPT-2 Small;
- 12 transformer-блоков;
- размерность residual stream: 768;
- библиотека TransformerLens для доступа к внутренним активациям;
- Sparse Autoencoder для поиска и оценки интерпретируемого semantic direction.

Основная точка вмешательства:

`blocks.6.hook_resid_pre`

Это соответствует входу в седьмой transformer-блок GPT-2.

---

## Steering direction

Для поиска steering-направления использовались SAE-признаки на седьмом блоке.

Кандидаты оценивались на двух наборах текстов:

- science-тексты;
- контрольные тексты.

Для каждого feature измерялась его активация на обоих типах данных.

Лучшим оказался SAE feature:

`17363`

На held-out validation он показал:

- science mean activation: 8.2169;
- control mean activation: 0.0;
- science active rate: 1.0;
- control active rate: 0.0.

Таким образом, этот feature хорошо отделяет science-тексты от контрольных и используется в дальнейшем как steering direction.

Steering vector определяется как соответствующий decoder vector SAE:

```math
v = W_{\mathrm{dec}}[17363]
```

Норма вектора близка к единице:

```math
\|v\|_2 \approx 1
```

Это позволяет интерпретировать $\alpha$ как приблизительную норму внесённой perturbation.

---

## Метрики

В работе используются несколько типов метрик.

### Concept score

После генерации текст повторно прогоняется через clean GPT-2 без steering и без denoiser.

На седьмом блоке вычисляются SAE activations, после чего берётся максимальная активация feature 17363 по токенам:

```math
C(x) = \max_t z_{t,17363}
```

Это позволяет оценить, насколько сам сгенерированный текст выражает целевой concept.

---

### Completion-only perplexity

Perplexity измеряется clean GPT-2 только на сгенерированном продолжении, без prompt-токенов.

```math
\mathrm{PPL}
=
\exp
\left(
-\frac{1}{N}
\sum_{i=1}^{N}
\log p(x_i \mid x_{<i})
\right)
```

Чем меньше PPL, тем более естественным для GPT-2 является полученный текст.

---

### Repetition metrics

Дополнительно измеряются:

- Dist-1;
- Dist-2;
- Dist-3;
- Rep-3.

Для $n$-грамм:

```math
\mathrm{dist}_n
=
\frac{
\#\{\text{unique n-grams}\}
}{
\#\{\text{all n-grams}\}
}
```

и

```math
\mathrm{rep}_3 = 1 - \mathrm{dist}_3
```

---

## Generation protocol

В основном steering evaluation используются:

- 6 нейтральных prompts;
- 5 random seeds;
- 50 новых токенов;
- temperature = 0.8;
- top-p = 0.95.

Все сравнения между raw steering и denoised steering выполняются при одинаковых параметрах генерации.

---

# Эксперименты

В ходе работы было исследовано несколько подходов.

---

## 1. Raw activation steering

Сначала был построен baseline без denoiser.

Интервенция выполнялась напрямую:

```math
h \rightarrow h + \alpha v
```

Основные результаты:

| alpha | Concept | PPL |
|---:|---:|---:|
| 0 | 3.2108 | 11.4722 |
| 4 | 3.7994 | 11.4190 |
| 8 | 4.7941 | 11.4439 |
| 16 | 7.1939 | 14.4280 |
| 32 | 7.7055 | 26.5748 |
| 64 | 0.0000 | 235.8455 |
| 96 | 0.0000 | 2890.6787 |
| 128 | 0.0000 | 19909.8083 |

Из этих результатов видно несколько режимов.

При $\alpha=4$ и $\alpha=8$ concept score растёт почти без ухудшения PPL.

При $\alpha=16$ semantic effect становится сильным, а PPL растёт умеренно.

При $\alpha=32$ concept остаётся высоким, но качество уже заметно ухудшается.

При больших значениях $\alpha$ происходит collapse: perplexity резко увеличивается, а целевой concept исчезает.

Практически полезный диапазон raw steering находится примерно между $\alpha=8$ и $\alpha=32$.

---

## 2. Same-layer Gaussian denoiser

Первый обучаемый baseline — обычный residual MLP, который восстанавливает noisy hidden states на том же слое.

Training corruption:

```math
x = h + \epsilon
```

где

```math
\epsilon \sim \mathcal N(0,\sigma^2 I)
```

Уровень шума задаётся через ожидаемую норму perturbation:

```math
\sigma = \frac{r}{\sqrt{768}}
```

Архитектура:

```math
(768 + 64)
\rightarrow
1536
\rightarrow
1536
\rightarrow
768
```

64 дополнительных координаты соответствуют embedding уровня шума.

Denoiser имеет residual-форму:

```math
D_\theta(x,\sigma)
=
x + \Delta_\theta(x,\sigma)
```

Loss — обычный reconstruction MSE:

```math
\mathcal L_{\mathrm{MSE}}
=
\mathbb E
\left[
\|D_\theta(h+\epsilon,\sigma)-h\|_2^2
\right]
```

Важно, что target steering direction при обучении не используется.

### Reconstruction results

На сильных noise levels модель действительно уменьшает reconstruction error.

Например:

| Noise norm | Raw MSE | Denoised MSE |
|---:|---:|---:|
| 32 | 1.3269 | 1.2710 |
| 64 | 5.3347 | 4.1166 |
| 96 | 12.0122 | 8.2564 |
| 128 | 21.2885 | 13.6826 |

Однако на слабых perturbations denoiser работал хуже или почти не давал выигрыша.

### Steering results

При $\alpha=16$ наблюдается небольшой matched-alpha improvement:

- concept: 7.1939 → 7.2884;
- PPL: 14.4280 → 14.3966.

Однако начиная примерно с $\alpha=32$ denoiser начинает резко ухудшать генерацию.

Этот эксперимент показывает, что хорошая reconstruction на random Gaussian noise не гарантирует хорошего переноса на structured semantic steering.

---

## 3. Downstream gated repair

В следующем эксперименте corruption вносился на седьмом блоке, а repair выполнялся уже на девятом.

Идея состояла в том, чтобы не пытаться восстановить исходную активацию напрямую, а исправлять последствия perturbation после того, как они распространились через несколько transformer-блоков.

Training corruption на седьмом блоке:

```math
h_7^{\mathrm{noisy}} = h_7 + \epsilon
```

На девятом блоке используется gated bottleneck.

Сначала входная активация вместе с embedding уровня шума проходит через encoder:

```math
z = E_\theta([x,\mathrm{emb}(\sigma)])
```

Архитектура bottleneck:

```math
(768+64) \rightarrow 1024 \rightarrow 256
```

После этого из latent representation вычисляется correction:

```math
\Delta h = H_{\mathrm{corr}}(z)
```

и gate:

```math
g = \mathrm{sigmoid}(H_{\mathrm{gate}}(z))
```

Итоговая repaired activation:

```math
R_\theta(x,\sigma) = x + g\Delta h
```

Gate должен позволять сети адаптивно выбирать силу вмешательства.

### Training objective

Использовалась комбинация нескольких loss terms:

```math
\mathcal L
=
\mathcal L_{\mathrm{LM}}
+
10^{-3}\mathcal L_{\mathrm{corr}}
+
0.5\mathcal L_{\mathrm{preserve}}
```

Для анализа downstream displacement вводились:

```math
d_{\mathrm{raw}}
=
h_9^{\mathrm{raw}}
-
h_9^{\mathrm{clean}}
```

и

```math
d_{\mathrm{rep}}
=
h_9^{\mathrm{repaired}}
-
h_9^{\mathrm{clean}}
```

Коэффициент сохранения направления:

```math
\rho
=
\frac{
\langle d_{\mathrm{rep}}, d_{\mathrm{raw}} \rangle
}{
\|d_{\mathrm{raw}}\|_2^2
}
```

Preservation penalty:

```math
\mathcal L_{\mathrm{preserve}}
=
\max(0,0.5-\rho)^2
```

### Training validation

На Gaussian validation результат выглядел очень сильным:

- clean PPL: 50.1114;
- noisy PPL: 90.4511;
- repaired PPL: 47.4812.

Средний gate:

```math
\bar g \approx 0.906
```

Коэффициент сохранения направления:

```math
\rho \approx 0.983
```

То есть repair не только восстанавливал качество после random noise, но и давал PPL ниже clean baseline.

### Failure mode

На target steering эксперимент оказался неудачным.

В рабочем диапазоне $\alpha$ concept score снижался, а PPL обычно рос.

Дополнительный анализ показал:

- gate почти всегда был близок к единице;
- correction norm была очень большой;
- сеть фактически превращалась в residual adapter, а не в минимальный repair.

Типичные значения correction norm находились примерно в диапазоне:

```math
\|\Delta h\|_2
\approx
30\text{--}60
```

Это важный отрицательный результат: хороший LM objective на training corruption сам по себе не гарантирует правильного поведения на semantic steering.

---

## 4. Naive interpolation denoiser

После Gaussian baseline была протестирована альтернативная corruption scheme:

```math
x_t
=
t h
+
(1-t)\epsilon
```

где

```math
\epsilon
\sim
\mathcal N(0,I)
```

В первой версии использовалось:

```math
t \sim U[0,1]
```

Архитектура denoiser оставалась residual:

```math
D_\theta(x_t,t)
=
x_t
+
\Delta_\theta(x_t,t)
```

На сильных corruption модель хорошо восстанавливала activations.

Например при $t=0.5$ reconstruction error уменьшался примерно в 10 раз.

Однако near-clean regime оказался нестабильным.

При $t\rightarrow1$ вход почти совпадает с clean hidden state, но сеть всё равно могла выдавать большую correction.

В предельном случае:

```math
t=1
```

и

```math
x_1=h
```

то есть raw reconstruction error равен нулю.

Однако denoiser сам сильно изменял clean activation.

Этот failure mode показал, что обычной residual parameterization недостаточно.

---

## 5. Improved interpolation denoiser

Финальный вариант исправляет проблему предыдущей модели двумя изменениями.

Во-первых, training range ограничивается:

```math
t \sim U[0.5,1]
```

Во-вторых, correction явно масштабируется через $1-t$:

```math
D_\theta(x_t,t)
=
x_t
+
(1-t)\Delta_\theta(x_t,t)
```

Это даёт важное свойство:

```math
D_\theta(h,1)
=
h
```

То есть при полностью clean input модель по конструкции не может вносить correction.

### Architecture

Используется token-wise MLP:

```math
(768+64)
\rightarrow
1536
\rightarrow
1536
\rightarrow
768
```

### Training corruption

```math
x_t
=
t h
+
(1-t)\epsilon
```

где

```math
\epsilon
\sim
\mathcal N(0,I)
```

и

```math
t
\sim
U[0.5,1]
```

### Reconstruction results

В отличие от naive interpolation version, improved architecture показывает стабильное улучшение во всём диапазоне.

| t | Raw MSE | Denoised MSE | Improvement |
|---:|---:|---:|---:|
| 0.99 | 0.010214 | 0.002717 | 3.759x |
| 0.95 | 0.255346 | 0.054729 | 4.666x |
| 0.90 | 1.021360 | 0.163536 | 6.245x |
| 0.80 | 4.084947 | 0.389751 | 10.481x |
| 0.70 | 9.190544 | 0.586672 | 15.666x |
| 0.60 | 16.341894 | 1.118401 | 14.612x |
| 0.50 | 25.532839 | 2.787411 | 9.160x |

Это показывает, что denoiser действительно научился восстанавливать activations на своей training corruption distribution.

---

## Отображение steering strength в denoiser conditioning

Training corruption и реальный steering имеют разную форму.

Training:

```math
x_t
=
t h
+
(1-t)\epsilon
```

Steering:

```math
h
+
\alpha v
```

Поэтому для conditioning denoiser используется fixed scale-matching heuristic.

Сначала вычисляется:

```math
q
=
\frac{
|\alpha|\|v\|_2
}{
\mathbb E\|h_7\|_2
}
```

После этого:

```math
t_\alpha
=
\frac{1}{1+q}
```

Значение дополнительно ограничивается training range.

---

# Main result

Главный положительный результат получен при:

```math
\alpha = 24
```

### Raw steering

Concept score:

```math
7.5621
```

PPL:

```math
22.2017
```

### Improved interpolation denoiser

Concept score:

```math
8.2392
```

PPL:

```math
21.4873
```

То есть одновременно:

```math
7.5621
\rightarrow
8.2392
```

по concept score и

```math
22.2017
\rightarrow
21.4873
```

по perplexity.

Это является локальным Pareto improvement относительно raw steering.

---

## Pareto interpretation

В данной задаче есть две цели:

```math
\text{Concept score} \uparrow
```

и

```math
\text{PPL} \downarrow
```

Одна точка Pareto-доминирует другую, если она не хуже по обеим метрикам и строго лучше хотя бы по одной.

Для $\alpha=24$ denoised point одновременно имеет:

- более высокий concept score;
- более низкий PPL.

Поэтому он строго доминирует соответствующий raw steering point.

---

# Основной исследовательский вывод

Эксперименты показывают, что capacity denoiser, вероятно, не является главным bottleneck.

Модели способны хорошо восстанавливать hidden states на тех perturbation distributions, на которых они обучались.

Проблема возникает при переносе на реальный activation steering.

Training perturbations в большинстве экспериментов являются случайными:

```math
\epsilon
\sim
\mathcal N(0,\sigma^2 I)
```

Steering perturbation имеет принципиально другую структуру:

```math
\delta h_{\mathrm{steer}}
=
\alpha v
```

Она полностью направлена вдоль одного семантически значимого направления.

Таким образом, основной limitation связан с mismatch между:

- training corruption;
- geometry of steering intervention;
- training objective;
- допустимой величиной repair correction.

---

## Почему improved interpolation работает лучше

Ключевое изменение:

```math
D_\theta(x_t,t)
=
x_t
+
(1-t)\Delta_\theta(x_t,t)
```

задаёт полезный inductive bias.

При:

```math
t\rightarrow1
```

получаем:

```math
1-t\rightarrow0
```

и correction автоматически уменьшается.

Это соответствует ожидаемому поведению:

- слабое повреждение → слабая correction;
- сильное повреждение → более сильная correction.

В предыдущих архитектурах такого ограничения не было, поэтому сеть могла слишком сильно изменять даже почти clean hidden states.

---

# Структура репозитория

Текущая структура проекта:

```text
.
├── README.md
├── figures/
├── report/
│   └── main.tex
└── src/
```

В `src/` находятся скрипты обучения и evaluation для основных экспериментов.

В `figures/` находятся графики результатов.

В `report/` находится полный исследовательский отчёт.

---

# Запуск

Рекомендуется использовать отдельное Python environment.

Основные зависимости:

```text
torch
transformer-lens
sae-lens
datasets
numpy
matplotlib
tqdm
```

Для создания окружения можно использовать:

```bash
python -m venv .venv
source .venv/bin/activate
```

Если в репозитории присутствует `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Обучение финального denoiser

Запуск training script improved interpolation model:

```bash
python src/tr_loss_um_1.py
```

После обучения создаётся checkpoint denoiser.

---

## Evaluation

Для оценки steering:

```bash
python src/test_improved_interpolation.py
```

Evaluation сравнивает:

- raw steering;
- steering + denoiser.

Основные метрики:

- concept score;
- completion-only PPL.

---

# Checkpoint

Лучший checkpoint публикуется отдельно в открытом репозитории на Hugging Face.

После загрузки здесь будет указана ссылка:

```text
https://huggingface.co/<username>/<model-repository>
```

---

# Отчёт

Полный исследовательский отчёт находится в:

```text
report/main.tex
```

Он содержит:

- постановку задачи;
- поиск SAE feature;
- raw steering baseline;
- Gaussian denoiser;
- downstream gated repair;
- naive interpolation experiment;
- improved interpolation denoiser;
- таблицы;
- графики;
- failure analysis;
- limitations;
- дальнейшие направления работы.

---

# Ограничения

У текущего evaluation есть несколько важных ограничений.

## Concept metric

Concept metric использует тот же SAE feature, который применяется как steering direction.

Это делает измерение интерпретируемым и удобным, однако в дальнейшем желательно добавить независимую semantic metric:

- отдельный classifier;
- LLM judge;
- другой SAE или probing model.

---

## Perplexity evaluator

PPL измеряется той же GPT-2 Small.

Это естественная мера совместимости с исходной моделью, однако независимая evaluator LM могла бы дать дополнительную проверку.

---

## Размер evaluation set

Evaluation включает:

- 6 prompts;
- 5 random seeds.

Для более статистически устойчивых выводов нужен более крупный benchmark.

---

## Distribution mismatch

Главное ограничение связано с тем, что training corruption не совпадает с реальным semantic steering.

Именно этот mismatch является одним из основных направлений дальнейшей работы.

---

# Возможные дальнейшие эксперименты

Перспективными выглядят следующие направления:

- обучение на смеси isotropic noise и случайных directional perturbations;
- ограничение относительной correction norm;
- более плотный sweep по $\alpha$ около диапазона 16–32;
- evaluation на нескольких SAE features;
- independent semantic classifier;
- feature-agnostic denoiser;
- более строгая оценка Pareto-front;
- alternative training objectives;
- исследование geometry downstream activations после steering;
- обучение repair model на perturbation distributions, более близких к semantic steering.

---

# Итог

В работе было показано, что learned denoising может улучшать activation steering, однако простой reconstruction objective не гарантирует хорошего переноса на semantic steering.

Same-layer Gaussian denoiser показал ограниченный эффект.

Downstream gated repair хорошо восстанавливал random-noise validation, но оказался слишком агрессивным на target steering.

Naive interpolation denoiser выявил проблему нестабильных corrections в near-clean regime.

Финальная improved interpolation architecture:

```math
D_\theta(x_t,t)
=
x_t
+
(1-t)\Delta_\theta(x_t,t)
```

устранила этот failure mode и дала локальный Pareto gain при:

```math
\alpha=24
```

где concept score увеличился:

```math
7.5621
\rightarrow
8.2392
```

а perplexity уменьшилась:

```math
22.2017
\rightarrow
21.4873
```

Главный вывод заключается в том, что основная проблема связана не столько с недостаточной мощностью denoiser, сколько с соответствием между training corruption, geometry of steering intervention и inductive bias repair-модели.

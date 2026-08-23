# T-Lab Interpretability — Improving Activation Steering with Learned Denoisers

Этот репозиторий содержит решение исследовательского задания по interpretability, посвящённого улучшению activation steering с помощью обучаемых denoiser / repair-модулей.

Основная идея работы заключается в следующем: стандартный activation steering позволяет усиливать выбранный семантический концепт в скрытых представлениях языковой модели, однако при слишком сильной интервенции качество генерации резко ухудшается. В этой работе исследуется, можно ли восстановить скрытые состояния после steering так, чтобы сохранить полезный семантический эффект и одновременно уменьшить деградацию качества текста.

В качестве базовой модели используется GPT-2 Small, а steering-направление извлекается из Sparse Autoencoder.

---

## Основная постановка

Стандартный activation steering модифицирует скрытое состояние модели следующим образом:

\[
\widetilde h = h + \alpha v,
\]

где:

- \(h\) — скрытое представление модели;
- \(v\) — steering direction;
- \(\alpha\) — сила интервенции.

При небольших значениях \(\alpha\) можно усилить нужный концепт практически без потери качества. Однако при увеличении \(\alpha\) активации начинают уходить из естественного распределения hidden states модели, что приводит к росту perplexity, ухудшению связности текста и, в предельном случае, к полному collapse генерации.

Цель работы — улучшить компромисс между двумя величинами:

- выраженностью целевого концепта;
- качеством генерации.

Иными словами, хочется получить одновременно более высокий concept score и более низкий perplexity.

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

Steering vector определяется как соответствующий decoder vector SAE.

Норма вектора близка к единице:

\[
\|v\|_2 \approx 1.
\]

Это позволяет интерпретировать \(\alpha\) как приблизительную норму внесённой perturbation.

---

## Метрики

В работе используются несколько типов метрик.

### Concept score

После генерации текст повторно прогоняется через clean GPT-2 без steering и без denoiser.

На седьмом блоке вычисляются SAE activations, после чего берётся максимальная активация feature 17363 по токенам.

Это позволяет оценить, насколько сам сгенерированный текст выражает целевой concept.

### Completion-only perplexity

Perplexity измеряется clean GPT-2 только на сгенерированном продолжении, без prompt-токенов.

Это основная метрика качества генерации.

Чем меньше PPL, тем более естественным для GPT-2 является полученный текст.

### Repetition metrics

Дополнительно измеряются:

- Dist-1;
- Dist-2;
- Dist-3;
- Rep-3.

Они используются для отслеживания повторов и деградации разнообразия текста.

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

\[
h \rightarrow h + \alpha v.
\]

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

При \(\alpha=4\) и \(\alpha=8\) concept score растёт почти бесплатно.

При \(\alpha=16\) semantic effect становится сильным, а PPL растёт умеренно.

При \(\alpha=32\) concept остаётся высоким, но качество уже заметно ухудшается.

При больших значениях \(\alpha\) происходит collapse: perplexity резко увеличивается, а целевой concept исчезает.

Практически полезный диапазон raw steering находится примерно между \(\alpha=8\) и \(\alpha=32\).

---

## 2. Same-layer Gaussian denoiser

Первый обучаемый baseline — обычный residual MLP, который восстанавливает noisy hidden states на том же слое.

Training corruption:

\[
x = h + \epsilon,
\]

где noise является изотропным Gaussian.

Архитектура:

\[
(768 + 64) \rightarrow 1536 \rightarrow 1536 \rightarrow 768.
\]

64 дополнительных координаты соответствуют embedding уровня шума.

Denoiser имеет residual-форму и предсказывает correction к входной активации.

Loss — обычный reconstruction MSE между восстановленной и clean activation.

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

При \(\alpha=16\) наблюдается небольшой matched-alpha improvement:

- concept: 7.1939 → 7.2884;
- PPL: 14.4280 → 14.3966.

Однако начиная примерно с \(\alpha=32\) denoiser начинает резко ухудшать генерацию.

Этот эксперимент показывает, что хорошая reconstruction на random Gaussian noise не гарантирует хорошего переноса на structured semantic steering.

---

## 3. Downstream gated repair

В следующем эксперименте corruption вносился на седьмом блоке, а repair выполнялся уже на девятом.

Идея состояла в том, чтобы не пытаться восстановить исходную активацию напрямую, а исправлять последствия perturbation после того, как они распространились через несколько transformer-блоков.

Архитектура представляет собой gated bottleneck.

Сначала входная активация вместе с embedding уровня шума проходит через encoder:

\[
(768 + 64) \rightarrow 1024 \rightarrow 256.
\]

После этого из bottleneck representation вычисляются:

- correction vector;
- scalar gate.

Итоговая repaired activation имеет вид:

\[
x_{\text{repaired}} = x + g \Delta h.
\]

Gate должен был позволить сети адаптивно выбирать силу вмешательства.

### Training objective

Использовалась комбинация:

- language modeling loss;
- correction penalty;
- preservation loss.

На Gaussian validation результат выглядел очень сильным:

- clean PPL: 50.1114;
- noisy PPL: 90.4511;
- repaired PPL: 47.4812.

То есть repair не только восстанавливал качество после noise, но и давал PPL ниже clean baseline.

### Failure mode

На target steering эксперимент оказался неудачным.

В рабочем диапазоне \(\alpha\) concept score снижался, а PPL обычно рос.

Дополнительный анализ показал:

- gate почти всегда был близок к единице;
- correction norm была очень большой;
- сеть фактически превращалась в residual adapter, а не в минимальный repair.

Это важный отрицательный результат: хороший LM objective на training corruption сам по себе не гарантирует правильного поведения на semantic steering.

---

## 4. Naive interpolation denoiser

После Gaussian baseline была протестирована альтернативная corruption scheme:

\[
x_t = t h + (1-t)\epsilon.
\]

В первой версии использовалось:

\[
t \sim U[0,1].
\]

Архитектура denoiser оставалась residual:

\[
D(x_t,t)=x_t+\Delta(x_t,t).
\]

На сильных corruption модель хорошо восстанавливала activations.

Например при \(t=0.5\) reconstruction error уменьшался примерно в 10 раз.

Однако near-clean regime оказался нестабильным.

При \(t\rightarrow1\) вход почти совпадает с clean hidden state, но сеть всё равно могла выдавать большую correction.

В предельном случае при \(t=1\) raw reconstruction error равен нулю, однако denoiser сам сильно портил активацию.

Этот failure mode показал, что обычной residual parameterization недостаточно.

---

## 5. Improved interpolation denoiser

Финальный вариант исправляет проблему предыдущей модели двумя изменениями.

Во-первых, training range ограничивается:

\[
t \sim U[0.5,1].
\]

Во-вторых, correction явно масштабируется через \(1-t\):

\[
D(x_t,t)=x_t+(1-t)\Delta(x_t,t).
\]

Это даёт важное свойство:

\[
D(h,1)=h.
\]

То есть при полностью clean input модель по конструкции не может вносить correction.

### Architecture

Используется token-wise MLP:

\[
(768+64)\rightarrow1536\rightarrow1536\rightarrow768.
\]

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

## Main result

Главный положительный результат получен при:

\[
\alpha = 24.
\]

Raw steering:

- Concept score: 7.5621;
- PPL: 22.2017.

Improved interpolation denoiser:

- Concept score: 8.2392;
- PPL: 21.4873.

То есть одновременно:

- concept усиливается;
- perplexity уменьшается.

Это является локальным Pareto improvement относительно raw steering.

Важно, что улучшение происходит не за счёт более крупной сети. Архитектура остаётся простой MLP. Основное изменение связано с corruption scheme и архитектурным inductive bias.

---

## Основной исследовательский вывод

Эксперименты показывают, что capacity denoiser не является главным bottleneck.

Модели способны хорошо восстанавливать hidden states на тех perturbation distributions, на которых они обучались.

Проблема возникает при переносе на реальный activation steering.

Training perturbations в большинстве экспериментов являются случайными:

\[
\epsilon \sim \mathcal N(0,\sigma^2 I).
\]

Steering perturbation имеет принципиально другую структуру:

\[
\delta h = \alpha v.
\]

Она полностью направлена вдоль одного семантически значимого направления.

Таким образом, основной limitation связан с mismatch между:

- training corruption;
- geometry of steering intervention;
- training objective;
- допустимой величиной repair correction.

---

## Почему improved interpolation работает лучше

Ключевое изменение:

\[
D(x_t,t)=x_t+(1-t)\Delta(x_t,t).
\]

Оно задаёт правильный inductive bias:

- слабое повреждение → слабая correction;
- сильное повреждение → более сильная correction.

При \(t\rightarrow1\) correction автоматически стремится к нулю.

В предыдущих архитектурах такого ограничения не было, поэтому сеть могла слишком сильно изменять даже почти clean hidden states.

---

# Структура репозитория

Примерная структура проекта:

```text
.
├── README.md
├── figures/
├── report/
│   └── main.tex
└── src/
    ├── train_gaussian_denoiser.py
    ├── test_gaussian_denoiser.py
    ├── train_gated_repair.py
    ├── test_gated_repair.py
    ├── train_naive_interpolation.py
    ├── test_naive_interpolation.py
    ├── train_improved_interpolation.py
    └── test_improved_interpolation.py

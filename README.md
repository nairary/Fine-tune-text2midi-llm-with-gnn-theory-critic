# Fine-tune text2midi LLM with GNN theory critic

Документация описывает полный путь от сырого HookTheory JSON до обученных GNN-моделей и inference. Базовая идея проекта:

- **TeacherGNN** учится на encoded HookTheory-песнях как theory critic: получает реальный и испорченный граф песни, восстанавливает замаскированные признаки и учится ранжировать real выше corrupted.
- **Chord scorer** восстанавливает аккордовые события из MIDI-сонорностей. Его обучаемые веса нужны observer-части, чтобы из MIDI построить теоретический граф.
- **ObserverGNN** учится дистиллировать скалярный score teacher-а, но уже по MIDI-derived graph. Это приближает teacher critic к сценарию, где на входе есть MIDI, а не исходный encoded JSON.

Все команды ниже предполагают запуск из корня репозитория. Если запускаешься в полностью новом окружении, иди сверху вниз: сначала окружение и sanity checks, потом подготовка данных, потом chord scorer, teacher, observer pipeline. Долгие GPU-запуски имеет смысл начинать только после коротких проверок из раздела 1.

## 0. Окружение

### 0.1 Клонирование и venv

```bash
git clone <repo-url>
cd Fine-tune-text2midi-llm-with-gnn-theory-critic

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Для локальных тестов дополнительно нужен `pytest`:

```bash
python -m pip install pytest
```

Быстрая проверка импортов:

```bash
python - <<'PY'
import torch
import torch_geometric
import pretty_midi
import hydra

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__)
print("pretty_midi", getattr(pretty_midi, "__version__", "unknown"))
print("hydra", hydra.__version__)
PY
```

Если используешь CUDA, проверь, что установленная сборка `torch` подходит под локальный драйвер. В примерах ниже `device=cpu` можно заменить на `device=cuda`, а для teacher лучше задавать оба поля:

```bash
device=cuda training.device=cuda
```

Рекомендуемый стиль запуска package entrypoint-ов:

```bash
python -m src.training.train_teacher
python -m src.observer.run_observer_pipeline
```

Для data-скриптов и scripts-утилит можно использовать `python -m ...` или прямой запуск файла.

## 1. Проверка новых изменений

Перед полной подготовкой и обучением можно быстро проверить, что свежие изменения по corruptions, staged SSL и pairwise MIDI/observer pipeline не сломаны.

```bash
python -m pytest -q \
  tests/test_song_corruptions_benign.py \
  tests/test_corruption_mode_balancer.py \
  tests/test_train_teacher_stages.py \
  tests/test_observer_offline_pipeline.py \
  tests/test_build_teacher_targets.py \
  tests/test_infer_observer_scores.py
```

Что покрывает этот набор:

- новые benign / near-benign corruptions и их metadata contract;
- балансировку corruption modes, чтобы редкие режимы реально доходили до датасета;
- двухстадийное обучение teacher-а: `mlm_ssl` -> `corruption`;
- cross-track ranking teacher-а: clean одного трека сравнивается не только со своим corrupted, но и с corrupted других треков в batch;
- offline observer pipeline: clean/corrupted пары рендерятся в отдельные MIDI, связываются через `*_pairs.jsonl`, получают teacher targets и загружаются как пары графов;
- batch inference observer-а для сравнения нескольких MIDI-кандидатов между собой.

После подготовки `teacher_encoded.json` из раздела 3 сделай еще короткий тренировочный smoke-run:

```bash
python -m src.training.train_teacher \
  +experiment=debug \
  data.json_path=data/HTCanon/encoded_full/teacher_encoded.json \
  dataloader.batch_size=2 \
  training.limit_train_samples=32 \
  training.limit_val_samples=16 \
  training.epochs=4 \
  experiment.epochs=4 \
  training.mlm_ssl_epochs=2 \
  training.corruption_epochs=2 \
  run_name=teacher_debug_staged
```

Успешный smoke-run должен создать `outputs/.../teacher_debug_staged/metrics.jsonl`, где первые строки имеют `stage=mlm_ssl`, а последующие `stage=corruption`. В corruption-stage смотри метрики `intra_rank_acc`, `inter_rank_acc`, `rank_acc`, `graph_binary_acc`: они подтверждают, что включены intra-track, cross-track и абсолютная clean/corrupted калибровка.

## 2. Ожидаемые входные данные

Минимальный набор HookTheory-данных:

```text
data/
  HookTheory/
    Hooktheory_Raw.json/
      4_merged.json
    HookTheoryStructure.train.jsonl
    HookTheoryStructure.val.jsonl
    HookTheoryStructure.test.jsonl
```

Опционально, если есть исходные MIDI:

```text
data/
  HookTheory/
    HookTheory_Train_MIDI/
    HookTheory_Val_MIDI/
    HookTheory_Test_MIDI/
```

Основной пайплайн не требует исходных MIDI: MIDI для chord scorer и observer можно сгенерировать из `teacher_encoded.json`.

В репозитории уже должны быть служебные словари и спецификации:

```text
metadata/
  specs/
  vocabs/
```

## 3. Работа с данными

Цель блока: получить `teacher_encoded.json`, который читает `HookTheoryDataset` и из которого строятся hetero-графы teacher-а.

### 3.1 Raw JSON -> processed JSON

```bash
python -m src.data.preprocess_hooktheory \
  --raw-json data/HookTheory/Hooktheory_Raw.json/4_merged.json \
  --out-dir data/HTCanon/HK_processed \
  --structure-train data/HookTheory/HookTheoryStructure.train.jsonl \
  --structure-val data/HookTheory/HookTheoryStructure.val.jsonl \
  --structure-test data/HookTheory/HookTheoryStructure.test.jsonl \
  --compute-stats
```

Скрипт сохраняет:

- `data/HTCanon/HK_processed/hooktheory_processed.json` - все распарсенные клипы;
- `data/HTCanon/HK_processed/hooktheory_processed_structured_only.json` - только клипы с найденными section-аннотациями;
- `data/HTCanon/HK_processed/hooktheory_processed.stats.json` - агрегаты;
- `hooktheory_processed_unmatched_ids.json` и `hooktheory_processed_unknown_split_ids.json` - диагностика несовпадений.

`valid` split нормализуется в `val`.

### 3.2 Таймлайны исходных песен

Этот шаг не обязателен для обучения teacher-а, но полезен для анализа section-структуры по `ori_uid`.

```bash
python -m src.data.build_preprocess_song_timelines \
  --processed-json data/HTCanon/HK_processed/hooktheory_processed.json \
  --out-dir data/HTCanon/HK_processed \
  --compute-stats
```

Выходы:

- `original_songs_timeline.json`;
- `original_songs_aggregate.stats.json`;
- `original_songs_timeline.stats.json`.

### 3.3 Processed JSON -> canonical JSON

```bash
python -m src.data.canonicalize_hooktheory \
  --input data/HTCanon/HK_processed/hooktheory_processed.json \
  --out-dir data/HTCanon/canonical_full
```

Для варианта только со structured-клипами:

```bash
python -m src.data.canonicalize_hooktheory \
  --input data/HTCanon/HK_processed/hooktheory_processed_structured_only.json \
  --out-dir data/HTCanon/canonical_structured_only
```

Выходы:

- `hooktheory_canonical.json`;
- `hooktheory_canonical.stats.json`;
- `hooktheory_canonical.report.json`.

Если нужна диагностика исходных сырых значений рядом с нормализованными, добавь `--keep-raw`.

### 3.4 Canonical JSON -> teacher encoded JSON

```bash
python -m src.data.encode_teacher_features \
  --input data/HTCanon/canonical_full/hooktheory_canonical.json \
  --metadata-dir metadata \
  --out-dir data/HTCanon/encoded_full
```

Для structured-only набора:

```bash
python -m src.data.encode_teacher_features \
  --input data/HTCanon/canonical_structured_only/hooktheory_canonical.json \
  --metadata-dir metadata \
  --out-dir data/HTCanon/encoded_structured_only
```

Выходы:

- `teacher_encoded.json` - главный датасет для teacher/observer pipeline;
- `teacher_encoded.stats.json`;
- `teacher_encoder_maps.json`.

Дефолтный teacher-конфиг ожидает:

```text
data/HTCanon/encoded_full/teacher_encoded.json
```

## 4. Обучение весов аккордового скорера

Chord scorer используется, когда observer строит аккордовые события из MIDI. Он берет MIDI-инструмент с именем `chords`, строит кандидаты аккордов по сонорностям и ранжирует их. Веса сохраняются в `learned_weights.yaml`.

### 4.1 Сгенерировать MIDI из encoded JSON

```bash
python -m src.data.render_encoded_song_to_midi \
  --encoded-json data/HTCanon/encoded_full/teacher_encoded.json \
  --output-root data/rendered \
  --overwrite
```

Структура выхода:

```text
data/rendered/
  train/*.mid
  val/*.mid
  test/*.mid
```

Каждый MIDI содержит два инструмента: `melody` и `chords`. Это важно: chord scorer по умолчанию ищет именно `chords`, а observer graph builder ищет `melody`.

Для быстрого smoke-run:

```bash
python -m src.data.render_encoded_song_to_midi \
  --encoded-json data/HTCanon/encoded_full/teacher_encoded.json \
  --output-root data/rendered_smoke \
  --split train \
  --limit 32 \
  --overwrite \
  --verbose
```

### 4.2 Обучить веса scorer-а

```bash
python scripts/fit_chord_score_weights.py \
  --encoded-json data/HTCanon/encoded_full/teacher_encoded.json \
  --midi-root data/rendered \
  --train-split train \
  --val-split val \
  --instrument-name chords \
  --epochs 200 \
  --lr 0.05 \
  --weight-decay 1e-4 \
  --chunk-size 16 \
  --eval-every 1 \
  --outdir outputs/chord_score_fit/full \
  --device cpu
```

Главные артефакты:

- `outputs/chord_score_fit/full/learned_weights.yaml` - веса для chord parser;
- `metrics.json` - итоговые метрики;
- `metrics.jsonl` - лог по эпохам;
- `last.pt` - checkpoint маленькой модели весов.

Если нужно отладить качество кандидатов, можно сохранить группы:

```bash
python scripts/fit_chord_score_weights.py \
  --encoded-json data/HTCanon/encoded_full/teacher_encoded.json \
  --midi-root data/rendered \
  --outdir outputs/chord_score_fit/debug \
  --limit-train 32 \
  --limit-val 32 \
  --epochs 20 \
  --save-train-groups-json \
  --save-val-groups-json
```

## 5. Обучение GNN Teacher

Teacher обучается на `teacher_encoded.json`. Он сам строит real/masked/corrupted графы, поэтому отдельные labels не нужны.
По умолчанию обучение теперь идет в 2 последовательных stage:

- `mlm_ssl`: только masked-reconstruction objective;
- `corruption`: graph-level ranking/calibration + local corruption objectives.

Если `training.mlm_ssl_epochs` и `training.corruption_epochs` не заданы, общее число эпох из `training.epochs` автоматически делится между stage примерно пополам. При необходимости разбиение можно задать явно.
Во втором stage teacher сравнивает не только clean/corrupted одной и той же песни, но и clean одной песни против corrupted других песен в том же batch. Дополнительно добавлена абсолютная калибровка `clean=1 / corrupted=0`, чтобы шкала score была согласованной между разными треками.

Практически это задается в `configs/config.yaml`:

- `losses.graph_rank_intra_weight=1.0` - clean_i должен быть выше corrupted_i;
- `losses.graph_rank_inter_weight=1.0` - clean_i должен быть выше corrupted_j для других песен в batch;
- `losses.graph_binary_weight=1.0` - clean получает label 1, corrupted получает label 0.

Для cross-track части нужен `dataloader.batch_size>=2`. Если поставить batch size 1, intra-track ranking останется, но inter-track сравнения между разными песнями в batch не будет.

Default full-набор song-level corruptions для teacher-а лежит в `configs/config.yaml` и сейчас включает:

```text
strongbeat_nonchord_note
borrowed_melody_conflict
borrowed_kind_toggle_without_melody_change
melody_semitone_add_clash
melody_suspension_clash
melody_alteration_clash
melody_omit_core_tone_conflict
inversion_bass_continuity_conflict
note_onset_shift
strong_weak_beat_flip
functional_progression_violation_strict
```

Benign / near-benign corruptions (`transpose_with_tonic_shift`, `merge_repeated_melody_notes`, `split_long_melody_note`, `melody_octave_shift`, `drop_tonic_seventh_on_strong_beat`) проверяются тестами из раздела 1 и могут запускаться явно через `dataloader.corruption_modes=[...]` или через `infer_teacher_score --modes ...`.

### 5.1 Smoke test

```bash
python -m src.training.train_teacher \
  +experiment=debug \
  data.json_path=data/HTCanon/encoded_full/teacher_encoded.json \
  dataloader.batch_size=2 \
  training.limit_train_samples=32 \
  training.limit_val_samples=16 \
  training.epochs=4 \
  experiment.epochs=4 \
  training.mlm_ssl_epochs=2 \
  training.corruption_epochs=2 \
  run_name=teacher_debug_staged
```

Проверь `outputs/.../teacher_debug_staged/metrics.jsonl`: должны быть строки со `stage=mlm_ssl`, затем со `stage=corruption`. В corruption-stage должны появиться `inter_rank_acc` и `inter_rank_loss`; это проверка сравнения между разными песнями в batch.

В текущей структуре `configs/config.yaml` уже развернут целиком, поэтому Hydra-группы подключаются через `+group=name`, например `+model=teacher_gnn_small`. Простые поля переопределяются обычным `a.b=value`.

### 5.2 Полное обучение

```bash
python -m src.training.train_teacher \
  data.json_path=data/HTCanon/encoded_full/teacher_encoded.json \
  dataloader.batch_size=32 \
  optimizer.lr=3e-4 \
  training.epochs=500 \
  experiment.epochs=500 \
  scheduler.t_max=500 \
  device=cuda \
  training.device=cuda \
  run_name=teacher_full_v1
```

Пример явного разбиения stage:

```bash
python -m src.training.train_teacher \
  training.epochs=500 \
  training.mlm_ssl_epochs=300 \
  training.corruption_epochs=200 \
  run_name=teacher_two_stage
```

Полезные варианты:

```bash
# меньшая модель
python -m src.training.train_teacher +model=teacher_gnn_small run_name=teacher_small

# structured-only датасет
python -m src.training.train_teacher \
  data.json_path=data/HTCanon/encoded_structured_only/teacher_encoded.json \
  run_name=teacher_structured_only

# OOD corruptions вместо default-набора
python -m src.training.train_teacher +dataloader=theory_aware_ood run_name=teacher_ood_modes
```

Hydra создает директорию запуска:

```text
outputs/YYYY-MM-DD/HH-MM-SS_<run_name>/
  composed_config.yaml
  run_metadata.json
  metrics.jsonl
  local_eval.json
  local_eval_examples.json
  checkpoints/
    last.pt
    best_recon_loss.pt
    best_rank_acc.pt
    mlm_ssl/
      last.pt
      best_recon_loss.pt
    corruption/
      last.pt
      best_rank_acc.pt
```

Для observer-а дальше нужны:

- teacher checkpoint: обычно `checkpoints/best_rank_acc.pt` или `checkpoints/last.pt`;
- matching config: `composed_config.yaml` из той же run-директории.

### 5.3 Оценка teacher-а

```bash
python -m src.training.eval_teacher_ssl \
  --checkpoint-path outputs/.../checkpoints/best_rank_acc.pt \
  data.json_path=data/HTCanon/encoded_full/teacher_encoded.json \
  device=cuda \
  training.device=cuda
```

OOD/benign corruption evaluation:

```bash
python scripts/eval_teacher_ood_corruptions.py \
  --dataset-json data/HTCanon/encoded_full/teacher_encoded.json \
  --checkpoint outputs/.../checkpoints/best_rank_acc.pt \
  --config outputs/.../composed_config.yaml \
  --split test \
  --mode-set ood \
  --device cuda \
  --outdir outputs/teacher_ood_eval
```

## 6. Кэширование MIDI-пар, teacher targets и graph cache

Observer pipeline строит пары `clean/corrupted`, рендерит их в MIDI, прогоняет teacher для получения target-score и кэширует observer-графы.

Здесь важно отличие от старого режима "сравнить только внутри одного encoded-трека": clean и corrupted сохраняются как отдельные MIDI-файлы, затем из каждого MIDI строится отдельный observer graph. Pair-связь хранится в `pairs/index/*_pairs.jsonl` и `targets/*_pairs.jsonl`; обучение observer-а загружает оба MIDI-derived графа и оптимизирует одновременно regression loss по teacher score и pair rank loss между clean/corrupted MIDI-графами.

Рекомендуемый единый запуск:

```bash
python -m src.observer.run_observer_pipeline \
  data.json_path=data/HTCanon/encoded_full/teacher_encoded.json \
  observer_pipeline.output_root=outputs/observer_pipeline_full \
  observer_training.teacher_checkpoint=outputs/.../checkpoints/best_rank_acc.pt \
  observer_training.teacher_config=outputs/.../composed_config.yaml \
  observer_training.chord_weights_yaml=outputs/chord_score_fit/full/learned_weights.yaml \
  observer_training.device=cuda \
  observer_training.epochs=20 \
  dataloader.batch_size=8 \
  optimizer.lr=1e-3
```

Что делает `run_observer_pipeline`:

1. `build_pairs`: создает clean/corrupted encoded JSON и MIDI.
2. `build_targets`: считает teacher score для каждого sample.
3. `build_graph_cache`: строит и сохраняет observer `HeteroData` графы.
4. `train`: обучает ObserverGNN на кэше.

Pairwise contract после шага `build_targets`:

- `targets/train.jsonl` и `targets/val.jsonl` - sample-level teacher scores для каждого clean/corrupted MIDI;
- `targets/train_pairs.jsonl` и `targets/val_pairs.jsonl` - clean/corrupted связи с `teacher_score_clean`, `teacher_score_corrupted`, `teacher_score_gap`;
- `training/metrics.jsonl` - observer metrics, включая `pair_rank_acc`, `mean_pred_margin`, `mean_teacher_margin`.

Структура выхода:

```text
outputs/observer_pipeline_full/
  pairs/
    encoded/train/*.json
    encoded/val/*.json
    midi/train/*.mid
    midi/val/*.mid
    manifests/train.jsonl
    manifests/val.jsonl
    index/train_pairs.jsonl
    index/val_pairs.jsonl
    skipped_manifest_rows.jsonl
  targets/
    train.jsonl
    val.jsonl
    train_pairs.jsonl
    val_pairs.jsonl
  cache/
    graphs/train/*.pt
    graphs/val/*.pt
    index/train.jsonl
    index/val.jsonl
  training/
    config.json
    metrics.jsonl
    best.pt
    last.pt
```

Если нужно выполнить только кэширование без обучения:

```bash
python -m src.observer.run_observer_pipeline \
  data.json_path=data/HTCanon/encoded_full/teacher_encoded.json \
  observer_pipeline.output_root=outputs/observer_cache_only \
  observer_training.teacher_checkpoint=outputs/.../checkpoints/best_rank_acc.pt \
  observer_training.teacher_config=outputs/.../composed_config.yaml \
  observer_training.chord_weights_yaml=outputs/chord_score_fit/full/learned_weights.yaml \
  observer_pipeline.train=false
```

Если пары и targets уже есть, а нужно только пересобрать графы и переобучить observer:

```bash
python -m src.observer.run_observer_pipeline \
  observer_pipeline.output_root=outputs/observer_pipeline_full \
  observer_pipeline.build_pairs=false \
  observer_pipeline.build_targets=false \
  observer_pipeline.build_graph_cache=true \
  observer_pipeline.train=true \
  observer_training.epochs=20
```

Важные настройки:

- `dataloader.pairs_per_song` - сколько corrupted-пар генерировать на одну песню;
- `dataloader.corruption_modes` - список song-level corruptions;
- `dataloader.theory_aware.deterministic_per_sample=true` нужен для воспроизводимых pair ids при `observer_pipeline.overwrite=false`;
- `observer_pipeline.overwrite=true` полностью пересобирает artifacts;
- `observer_training.chord_weights_yaml=null` включает ручной scorer вместо обученных весов;
- `observer_training.chord_instrument_name=chords` задает MIDI-инструмент для гармонического анализа;
- `losses.use_pair_rank=true` включает pair rank loss;
- `losses.min_teacher_gap_for_rank=0.25` отбрасывает слишком неоднозначные пары из rank loss, но regression loss все равно считается для clean и corrupted.

## 7. Обучение GNN Observer

Если используешь единый `run_observer_pipeline`, обучение запускается последним шагом автоматически.

Для отдельного запуска обучения по уже готовому cache:

```bash
python -m src.observer.run_observer_pipeline \
  observer_pipeline.output_root=outputs/observer_pipeline_full \
  observer_pipeline.build_pairs=false \
  observer_pipeline.build_targets=false \
  observer_pipeline.build_graph_cache=false \
  observer_pipeline.train=true \
  observer_training.epochs=20 \
  observer_training.device=cuda \
  dataloader.batch_size=8
```

Observer обучается на pair loss:

- regression loss: предсказывает teacher score для clean и corrupted;
- rank loss: учится сохранять порядок teacher-а внутри пары, если teacher gap достаточно большой.

Ключевые метрики в `training/metrics.jsonl`:

- `loss`, `reg_loss`, `rank_loss`;
- `mae`, `rmse`;
- `pearson`, `spearman`;
- `pair_rank_acc`;
- `mean_pred_margin`, `mean_teacher_margin`.

Resume:

```bash
python -m src.observer.run_observer_pipeline \
  observer_pipeline.output_root=outputs/observer_pipeline_full \
  observer_pipeline.build_pairs=false \
  observer_pipeline.build_targets=false \
  observer_pipeline.build_graph_cache=false \
  observer_pipeline.train=true \
  observer_training.resume=true \
  observer_training.epochs=40
```

`observer_training.epochs` должен быть больше эпохи в `training/last.pt`.

## 8. Inference

### 8.1 Teacher score для одного encoded song

`infer_teacher_score` ожидает один JSON-объект песни, а не весь `{song_id: song}` датасет. Если нужно, сначала сохрани одну песню из `teacher_encoded.json` отдельным файлом.

```bash
python -m src.inference.infer_teacher_score \
  --encoded-json data/tmp/song.json \
  --checkpoint outputs/.../checkpoints/best_rank_acc.pt \
  --config outputs/.../composed_config.yaml \
  --device cuda \
  --pretty
```

С применением corruptions:

```bash
python -m src.inference.infer_teacher_score \
  --encoded-json data/tmp/song.json \
  --checkpoint outputs/.../checkpoints/best_rank_acc.pt \
  --config outputs/.../composed_config.yaml \
  --backend song_theory \
  --modes strongbeat_nonchord_note note_onset_shift \
  --save-corrupted-json tmp/song_corrupted.json \
  --pretty
```

Выход содержит `original_score`, `corrupted_score` и `score_gap`.

### 8.2 Аккорды из MIDI

```bash
python scripts/predict_midi_chords.py \
  --midi-path data/rendered/val/<song_id>.mid \
  --tonic-pc 0 \
  --mode major \
  --instrument-name chords \
  --weights-yaml outputs/chord_score_fit/full/learned_weights.yaml \
  --json-out outputs/chord_preds/<song_id>.json \
  --pretty
```

`--tonic-pc`: pitch class тоники, `C=0`, `C#/Db=1`, ..., `B=11`.

`--mode`: один из `major`, `minor`, `dorian`, `phrygian`, `lydian`, `mixolydian`, `locrian`, `harmonic_minor`, `phrygian_dominant`.

### 8.3 Batch observer score для GRPO

Для GRPO reward/scoring есть отдельный batch CLI:

```bash
python -m src.inference.infer_observer_scores \
  --input-json data/tmp/grpo_candidates.json \
  --checkpoint outputs/observer_pipeline_full/training/best.pt \
  --output-json outputs/grpo_scores.json \
  --device cuda \
  --batch-size 8 \
  --pretty
```

Формат входа - JSON-массив объектов. Поддерживаются поля из запроса с текущими опечатками `meter_numenator` / `meter_denumenator`; правильные варианты `meter_numerator` / `meter_denominator` тоже принимаются.

```json
[
  {
    "midi_path": "outputs/grpo/candidate_001.mid",
    "key": "C",
    "mode": "major",
    "bpm": 120,
    "meter_numenator": 4,
    "meter_denumenator": 4
  },
  {
    "midi_path": "outputs/grpo/candidate_002.mid",
    "key": "Bb",
    "mode": "minor",
    "bpm": 96,
    "meter_numerator": 6,
    "meter_denominator": 8
  }
]
```

MIDI должен содержать non-drum инструменты `melody` и `chords`. Если гармонический инструмент называется иначе:

```bash
python -m src.inference.infer_observer_scores \
  --input-json data/tmp/grpo_candidates.json \
  --checkpoint outputs/observer_pipeline_full/training/best.pt \
  --chord-instrument-name harmony \
  --output-json outputs/grpo_scores.json
```

Выход сохраняет соответствие с входным порядком в поле `scores`, а также возвращает подробные `results` с `index`, `score` и `rank`:

```json
{
  "scores": [0.42, 0.37],
  "results": [
    {"index": 0, "midi_path": "outputs/grpo/candidate_001.mid", "score": 0.42, "rank": 1},
    {"index": 1, "midi_path": "outputs/grpo/candidate_002.mid", "score": 0.37, "rank": 2}
  ]
}
```

Если GRPO-коду удобнее получить `results` сразу отсортированными по убыванию score, добавь `--sort`. Поле `scores` все равно остается aligned по исходному порядку.

По умолчанию CLI берет `chord_weights_yaml`, `chord_instrument_name` и `use_fallback_44` из config, сохраненного внутри observer checkpoint. Любой из этих параметров можно переопределить CLI-аргументом.

## 9. Быстрый end-to-end чеклист

Это минимальный порядок команд для нового окружения. Если GPU нет, замени `device=cuda training.device=cuda observer_training.device=cuda` на `cpu`.

```bash
# 0. окружение
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest

# 1. короткие проверки новых corruption / SSL / pairwise observer изменений
python -m pytest -q \
  tests/test_song_corruptions_benign.py \
  tests/test_corruption_mode_balancer.py \
  tests/test_train_teacher_stages.py \
  tests/test_observer_offline_pipeline.py \
  tests/test_build_teacher_targets.py \
  tests/test_infer_observer_scores.py

# 2. raw -> processed
python -m src.data.preprocess_hooktheory \
  --raw-json data/HookTheory/Hooktheory_Raw.json/4_merged.json \
  --out-dir data/HTCanon/HK_processed \
  --structure-train data/HookTheory/HookTheoryStructure.train.jsonl \
  --structure-val data/HookTheory/HookTheoryStructure.val.jsonl \
  --structure-test data/HookTheory/HookTheoryStructure.test.jsonl \
  --compute-stats

# 3. processed -> canonical -> encoded
python -m src.data.canonicalize_hooktheory \
  --input data/HTCanon/HK_processed/hooktheory_processed.json \
  --out-dir data/HTCanon/canonical_full

python -m src.data.encode_teacher_features \
  --input data/HTCanon/canonical_full/hooktheory_canonical.json \
  --metadata-dir metadata \
  --out-dir data/HTCanon/encoded_full

# 4. teacher staged smoke-run
python -m src.training.train_teacher \
  +experiment=debug \
  data.json_path=data/HTCanon/encoded_full/teacher_encoded.json \
  dataloader.batch_size=2 \
  training.limit_train_samples=32 \
  training.limit_val_samples=16 \
  training.epochs=4 \
  experiment.epochs=4 \
  training.mlm_ssl_epochs=2 \
  training.corruption_epochs=2 \
  run_name=teacher_debug_staged

# 5. MIDI + chord scorer
python -m src.data.render_encoded_song_to_midi \
  --encoded-json data/HTCanon/encoded_full/teacher_encoded.json \
  --output-root data/rendered \
  --overwrite

python scripts/fit_chord_score_weights.py \
  --encoded-json data/HTCanon/encoded_full/teacher_encoded.json \
  --midi-root data/rendered \
  --outdir outputs/chord_score_fit/full \
  --epochs 200 \
  --device cpu

# 6. full teacher
python -m src.training.train_teacher \
  data.json_path=data/HTCanon/encoded_full/teacher_encoded.json \
  dataloader.batch_size=32 \
  training.epochs=500 \
  experiment.epochs=500 \
  scheduler.t_max=500 \
  device=cuda \
  training.device=cuda \
  run_name=teacher_full_v1

# 7. выбери fresh teacher run
TEACHER_RUN=$(ls -td outputs/*/*teacher_full_v1* | head -n 1)
echo "$TEACHER_RUN"

# 8. teacher eval
python -m src.training.eval_teacher_ssl \
  --checkpoint-path "$TEACHER_RUN/checkpoints/best_rank_acc.pt" \
  data.json_path=data/HTCanon/encoded_full/teacher_encoded.json \
  device=cuda \
  training.device=cuda

# 9. observer MIDI-pair cache + targets + train
python -m src.observer.run_observer_pipeline \
  data.json_path=data/HTCanon/encoded_full/teacher_encoded.json \
  observer_pipeline.output_root=outputs/observer_pipeline_full \
  observer_training.teacher_checkpoint="$TEACHER_RUN/checkpoints/best_rank_acc.pt" \
  observer_training.teacher_config="$TEACHER_RUN/composed_config.yaml" \
  observer_training.chord_weights_yaml=outputs/chord_score_fit/full/learned_weights.yaml \
  observer_training.device=cuda \
  observer_training.epochs=20 \
  dataloader.batch_size=8 \
  optimizer.lr=1e-3
```

Контроль после полного observer run:

```bash
tail -n 1 outputs/observer_pipeline_full/training/metrics.jsonl
wc -l outputs/observer_pipeline_full/targets/train_pairs.jsonl
wc -l outputs/observer_pipeline_full/targets/val_pairs.jsonl
```

В последней строке `metrics.jsonl` смотри `val.pair_rank_acc`, `val.mae`, `val.spearman`, `val.mean_pred_margin`. Ненулевые `*_pairs.jsonl` подтверждают, что observer учился на MIDI-парах, а не на одиночных samples.

После этого основные production-артефакты:

- Teacher checkpoint: `outputs/.../checkpoints/best_rank_acc.pt`;
- Teacher config: `outputs/.../composed_config.yaml`;
- Chord scorer weights: `outputs/chord_score_fit/full/learned_weights.yaml`;
- Observer checkpoint: `outputs/observer_pipeline_full/training/best.pt`.

## 10. Частые проблемы

**`Instrument 'chords' not found`**

MIDI не содержит non-drum track с именем `chords`. Используй `render_encoded_song_to_midi`, переименуй track или передай другой `--instrument-name` / `observer_training.chord_instrument_name`.

**`Instrument 'melody' not found`**

Observer graph builder сейчас ожидает melody track с именем `melody`.

**`No train pairs were built`**

Проверь, что encoded songs содержат `meta.split`, `main_key_tonic_pc`, `main_key_scale_id`, `main_bpm`, `main_num_beats`, `main_beat_unit`, а выбранные corruptions реально применяются. Детали лежат в `pairs/skipped_manifest_rows.jsonl`.

**Teacher target bootstrap падает на первом sample**

`observer_training.teacher_checkpoint` и `observer_training.teacher_config` должны быть от одного и того же teacher run. Архитектура в config должна совпадать с checkpoint.

**Hydra override `model=teacher_gnn_small` не работает**

В текущем базовом `configs/config.yaml` нет defaults-list, поэтому для config group используй `+model=teacher_gnn_small`, `+experiment=debug`, `+dataloader=theory_aware_ood`. Для обычных полей используй `model.hidden_dim=...`, `optimizer.lr=...`.

**Запуск из корневого `train_teacher.py` падает на configs**

Используй package entrypoint:

```bash
python -m src.training.train_teacher
```

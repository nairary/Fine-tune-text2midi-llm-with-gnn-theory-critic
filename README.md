# Teacher SSL training with Hydra

## What changed

The teacher SSL training stack now uses Hydra composition as a single configuration entrypoint. You can switch model, data, dataloader, optimizer, scheduler, losses, and experiment settings from YAML config groups or via CLI overrides without editing Python code.

## Config layout

```text
configs/
  config.yaml
  model/
  data/
  dataloader/
  training/
  optimizer/
  scheduler/
  losses/
  experiment/
  hydra/
```

Available config groups:

- `model`
- `data`
- `dataloader`
- `training`
- `optimizer`
- `scheduler`
- `losses`
- `experiment`
- `hydra`

## Main entrypoint

Use either of these equivalent commands:

```bash
python train_teacher.py
python src/training/train_teacher.py
```

Hydra will create a separate run directory under `outputs/` for single runs and `multirun/` for sweeps, and it will persist the rendered config in `.hydra/` as well as `composed_config.yaml`.

## Examples

### Baseline run

```bash
python train_teacher.py
```

### Override batch size / learning rate / epochs

```bash
python train_teacher.py dataloader.batch_size=4 optimizer.lr=3e-4 training.epochs=5
```

### Switch model preset

```bash
python train_teacher.py model=teacher_gnn_small
```

### Switch dataset preset

```bash
python train_teacher.py data=teacher_encoded_structured_only
```

### Run debug experiment preset

```bash
python train_teacher.py experiment=debug
```

### Multirun / sweep

```bash
python train_teacher.py --multirun optimizer.lr=1e-3,3e-4,1e-4 training.seed=1,2,3
```

## Outputs saved per run

Each Hydra run directory stores:

- the composed Hydra config
- `composed_config.yaml`
- `metrics.jsonl`
- `run_metadata.json`
- checkpoints in `checkpoints/`

## Notes

- `data=teacher_encoded_structured_only` expects `data/HTCanon/encoded_structured_only/teacher_encoded.json` to exist.
- Relative dataset and metadata paths are resolved against the original project root, even though Hydra changes the working directory for each run.

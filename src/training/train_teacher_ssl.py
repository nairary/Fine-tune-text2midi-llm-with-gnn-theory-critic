from __future__ import annotations

from src.training.train_teacher import (
    SplitFilteredDataset,
    build_loaders,
    build_model,
    evaluate,
    main,
    print_metrics,
    run_epoch,
    save_checkpoint,
    set_seed,
)


if __name__ == "__main__":
    main()

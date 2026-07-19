# Run logging kept out of the AL loop: prints a console line per round and, if
# enabled, mirrors the SAME scalars to wandb -- accuracy + the regular AL metrics
# (labels-per-class, num-labeled, coverage). Nothing else goes to wandb.

try:
    import wandb
except ImportError:
    wandb = None # only used for --wandb runs; absent in .venv_clean


class RunLogger:

    def __init__(self, use_wandb=False, wandb_kwargs=None, acc_metric="test_accuracy"):
        self.use_wandb = bool(use_wandb) and wandb is not None
        self.acc_metric = acc_metric                    # fine accuracy field name (a wandb y-axis)
        self.coarse_metric = "test_coarse_accuracy"     # optional superclass-accuracy field name
        if self.use_wandb:
            wandb.init(**(wandb_kwargs or {}))
            # both accuracies plot against labels-per-class; the coarse one only appears when logged
            wandb.define_metric(self.acc_metric, step_metric="budget_per_class")
            wandb.define_metric(self.coarse_metric, step_metric="budget_per_class")

    def set_config(self, **values):
        if self.use_wandb:
            wandb.config.update(values, allow_val_change=True) # e.g. the resolved delta

    def log_round(self, r, num_labeled, budget_per_class, accuracy, coverage=None, coarse_accuracy=None):
        # console line for the slurm log
        msg = f"round {r}: labeled={num_labeled} ({budget_per_class:.1f}B) {self.acc_metric}={accuracy:.2f}"
        if coarse_accuracy is not None:
            msg += f" coarse={coarse_accuracy:.2f}"
        if coverage is not None:
            msg += f" coverage={coverage:.3f}"
        print(msg, flush=True)
        # the only things sent to wandb: the accuracy (+ optional coarse accuracy) + the regular scalars
        if self.use_wandb:
            metrics = {"budget_per_class": budget_per_class, "num_labeled": num_labeled,
                       self.acc_metric: accuracy, "episode": r}
            if coarse_accuracy is not None:
                metrics[self.coarse_metric] = coarse_accuracy
            if coverage is not None:
                metrics["coverage"] = coverage
            wandb.log(metrics)

    def finish(self):
        if self.use_wandb:
            wandb.finish()

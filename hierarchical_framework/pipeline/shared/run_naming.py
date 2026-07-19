# One place to turn an embeddings path into the names a run's outputs use, so the
# runner and the analysis tools all name things the same way.
#
# The embeddings live at embeddings/<dataset>/<technique>/<source>, so the last three
# path parts are exactly (dataset, technique, source).

from pathlib import Path


def dataset_technique_source(embeddings_dir):
    # e.g. .../embeddings/cifar100/simclr/old_deep_al -> ("cifar100", "simclr", "old_deep_al")
    return Path(embeddings_dir).parts[-3:]


def figure_output_base(embeddings_dir, delta):
    # Nested path used for figure folders / filenames, e.g. cifar100/simclr/old_deep_al_d0.65.
    # Strategy and budget are left out on purpose -- they are defaults kept in the wandb config.
    dataset, technique, source = dataset_technique_source(embeddings_dir)
    return f"{dataset}/{technique}/{source}_d{delta}"


def wandb_run_tag(embeddings_dir):
    # Flat tag for a wandb run name, e.g. cifar100_simclr_old_deep_al.
    return "_".join(dataset_technique_source(embeddings_dir))

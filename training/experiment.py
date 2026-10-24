from pathlib import Path


def experiment_paths(root, setting, training_iter=0):
    directory = Path(root) / setting / str(training_iter)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def results_paths(root, setting, training_iter=0):
    return experiment_paths(root, setting, training_iter)

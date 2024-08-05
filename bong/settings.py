from pathlib import Path

# Hyperparameters
hparam_path = Path(Path(__file__).resolve().parent, "hyperparameters")
hparam_mnist_path = Path(hparam_path, "mnist_hyperparameters")

# Result
result_path = Path(Path(__file__).resolve().parent, "results")
linreg_path = Path(result_path, "linreg_results")
logreg_path = Path(result_path, "logreg_results")
uci_path = Path(result_path, "uci_results")
mnist_path = Path(result_path, "mnist_results")

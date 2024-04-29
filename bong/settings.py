from pathlib import Path

# Result
result_path = Path(Path(__file__).resolve().parent, "results")
linreg_path = Path(result_path, "linreg_results")
logreg_path = Path(result_path, "logreg_results")
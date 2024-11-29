import os
from rafale.run import TrainingRun

ENV_VARS = {key: value for key, value in os.environ.items()}

def save_output_run_status(status: int, output_dir: str):
    with open(os.path.join(output_dir, "run.out"), "w") as f:
        f.write(str(status))

def check_run_status(output_dir: str) -> int:
    try:
        with open(os.path.join(output_dir, "run.out"), "r") as f:
            content = f.read().strip()  # Read and strip whitespace
            if content in {"0", "1"}:  # Check if content is '0' or '1'
                return int(content)
            else:
                raise ValueError("Invalid content in run.out. Expected '0' or '1'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"run.out not found in directory: {output_dir}")

def main():
    train_run = TrainingRun()
    train_run()

if __name__ == "__main__":
    main()

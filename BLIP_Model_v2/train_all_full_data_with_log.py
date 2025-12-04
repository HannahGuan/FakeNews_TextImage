"""
Wrapper script to run training with output logging
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_with_logging():
    # Create logs directory
    log_dir = Path("training_logs")
    log_dir.mkdir(exist_ok=True)

    # Generate timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = log_dir / f"full_training_{timestamp}.txt"

    print("="*70)
    print("Training BLIP Model v2 with Full Training Data")
    print("Using Best Hyperparameters from Grid Search")
    print("="*70)
    print(f"\nLog file: {logfile}\n")

    # Training configurations
    configs = [
        ("6-way", "6_way"),
        ("3-way", "3_way"),
        ("2-way", "2_way"),
    ]

    # Best hyperparameters from grid search
    params = [
        "--mode", "train",
        "--batch-size", "8",
        "--lr", "0.0001",
        "--dropout", "0.3",
        "--pooling", "mean",
        "--epochs", "10",
        "--eval-after-train"
    ]

    with open(logfile, 'w', encoding='utf-8') as log_f:
        log_f.write("="*70 + "\n")
        log_f.write("Training BLIP Model v2 with Full Training Data\n")
        log_f.write("Using Best Hyperparameters from Grid Search\n")
        log_f.write("="*70 + "\n")
        log_f.write(f"Started at: {datetime.now()}\n\n")

        for idx, (name, type_code) in enumerate(configs, 1):
            print(f"\n{'='*70}")
            print(f"[{idx}/3] Training {name} Classification")
            print(f"{'='*70}\n")

            log_f.write(f"\n{'='*70}\n")
            log_f.write(f"[{idx}/3] Training {name} Classification\n")
            log_f.write(f"{'='*70}\n\n")
            log_f.flush()

            # Build command
            cmd = ["python", "-u", "main.py", "--classification-type", type_code] + params

            # Run training with real-time output to both terminal and file
            # -u flag forces unbuffered output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )

            # Stream output to both terminal and log file
            for line in process.stdout:
                print(line, end='')
                log_f.write(line)
                log_f.flush()

            process.wait()

            if process.returncode != 0:
                error_msg = f"\nERROR: {name} training failed with return code {process.returncode}!"
                print(error_msg)
                log_f.write(error_msg + "\n")
                sys.exit(process.returncode)

            success_msg = f"\n[{idx}/3] {name} classification complete!\n"
            print(success_msg)
            log_f.write(success_msg + "\n")

        # Final summary
        log_f.write(f"\n{'='*70}\n")
        log_f.write("ALL TRAINING COMPLETE!\n")
        log_f.write(f"{'='*70}\n")
        log_f.write(f"Finished at: {datetime.now()}\n\n")
        log_f.write("Trained models saved to:\n")
        log_f.write("  - checkpoints/6_way/best_model.pt\n")
        log_f.write("  - checkpoints/3_way/best_model.pt\n")
        log_f.write("  - checkpoints/2_way/best_model.pt\n\n")
        log_f.write("Evaluation results saved to:\n")
        log_f.write("  - logs/6_way/\n")
        log_f.write("  - logs/3_way/\n")
        log_f.write("  - logs/2_way/\n")

    print(f"\n{'='*70}")
    print("ALL TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nFull log saved to: {logfile}\n")
    print("Trained models saved to:")
    print("  - checkpoints/6_way/best_model.pt")
    print("  - checkpoints/3_way/best_model.pt")
    print("  - checkpoints/2_way/best_model.pt\n")
    print("Evaluation results saved to:")
    print("  - logs/6_way/")
    print("  - logs/3_way/")
    print("  - logs/2_way/\n")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        run_with_logging()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

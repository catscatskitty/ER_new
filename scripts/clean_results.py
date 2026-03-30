#!/usr/bin/env python
import shutil
from pathlib import Path


def clean_results(keep_last=True):
    """Очистка результатов (сохраняет только последние)"""
    results_root = Path('results')
    
    # Очистка manual_checks
    manual_dir = results_root / 'manual_checks'
    if manual_dir.exists():
        files = list(manual_dir.glob('batch_*.csv'))
        if keep_last and files:
            files.sort(key=lambda x: x.stat().st_mtime)
            for f in files[:-1]:
                f.unlink()
                print(f"Deleted: {f}")
    
    # Очистка metrics
    metrics_dir = results_root / 'metrics'
    if metrics_dir.exists():
        for pattern in ['*_metrics.json', '*_history.json', '*_comparison_*.csv']:
            files = list(metrics_dir.glob(pattern))
            if keep_last and files:
                files.sort(key=lambda x: x.stat().st_mtime)
                for f in files[:-1]:
                    f.unlink()
                    print(f"Deleted: {f}")
    
    print("Cleanup completed!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep-last', action='store_true', default=True)
    parser.add_argument('--delete-all', action='store_true')
    args = parser.parse_args()
    
    clean_results(keep_last=not args.delete_all)


if __name__ == "__main__":
    main()
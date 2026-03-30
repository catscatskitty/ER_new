#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path

def main():
    print("="*60)
    print("Running Full Pipeline")
    print("="*60)
    
    steps = [
        ("Extracting features", "scripts/extract_all_features.py"),
        ("Training models", "src/training/train_pipeline.py"),
        ("Evaluating models", "src/evaluation/compare_models.py"),
    ]
    
    for name, script in steps:
        print(f"\n[{name}]")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"❌ {name} failed!")
            return
    
    print("\n✅ Pipeline completed!")
    print("Launch Streamlit: streamlit run src/manual_check/app.py")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.gpu_utils import check_gpu, print_gpu_info


def main():
    print("="*50)
    print("GPU Information")
    print("="*50)
    
    if check_gpu():
        print_gpu_info()
    else:
        print("GPU not available. Using CPU.")


if __name__ == "__main__":
    main()
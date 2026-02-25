"""
gpu_stress_test.py

A simple utility script to verify PyTorch CUDA installation and ensure
the GPU can handle sustained high-load matrix multiplications.
"""

import torch

def run_stress_test():
    """Saturates the GPU with large matrix multiplications."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing stress test on compute device: {device}")

    if device.type == 'cpu':
        print("Warning: CUDA is not available. This will run slowly on CPU.")

    # Create large random tensors
    matrix_size = (8192, 8192)
    print(f"Allocating {matrix_size} tensors in VRAM...")

    a = torch.randn(matrix_size, device=device)
    b = torch.randn(matrix_size, device=device)

    print("Stressing GPU... Press Ctrl+C to stop.")

    try:
        iteration = 0
        while True:
            # Heavy matrix multiplication
            c = torch.matmul(a, b)
            c = torch.relu(c)

            # Free memory immediately to prevent OOM build up
            del c
            torch.cuda.synchronize()

            iteration += 1
            if iteration % 100 == 0:
                print(f"Completed {iteration} iterations successfully.")

    except KeyboardInterrupt:
        print("\nStress test stopped cleanly by user.")

if __name__ == "__main__":
    run_stress_test()
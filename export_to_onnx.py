"""
export_to_onnx.py

Converts the trained PyTorch GPT model into ONNX format for deployment
and use with OpenVINO / NPU accelerated runtimes.
"""

import os
import torch
from dotenv import load_dotenv

# Import architecture definition from the main inference script
from chat_gpt_bpe import GPTLanguageModel, MODEL_PATH, BLOCK_SIZE, device

load_dotenv()
ONNX_OUTPUT_PATH = os.getenv("ONNX_MODEL_PATH", "gpt_model.onnx")

def export_model():
    """Loads PyTorch state dictionary and exports static ONNX computation graph."""
    print(f"Loading PyTorch model from {MODEL_PATH}...")
    model = GPTLanguageModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Create dummy input tensor to trace the model graph
    dummy_input = torch.randint(0, 100, (1, BLOCK_SIZE), dtype=torch.long, device=device)

    print(f"Exporting ONNX model to {ONNX_OUTPUT_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {1: "seq_len"},
            "output": {1: "seq_len"}
        }
    )
    print("Export Complete.")

if __name__ == "__main__":
    export_model()
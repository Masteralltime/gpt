"""
chat_openvino_npu.py

Loads an ONNX-exported GPT model using the OpenVINO runtime, forcing execution
onto a Neural Processing Unit (NPU) for high-efficiency, low-power inference.
"""

import os
import re
import numpy as np
from tokenizers import Tokenizer
from openvino.runtime import Core
from dotenv import load_dotenv

load_dotenv()
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer.json")
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "gpt_model.onnx")
BLOCK_SIZE = 256

# Verify files exist
if not os.path.exists(ONNX_MODEL_PATH):
    raise FileNotFoundError(f"Cannot find ONNX model at {ONNX_MODEL_PATH}. Export the model first.")

# Initialize Tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
def encode(text): return tokenizer.encode(text).ids
def decode(token_ids): return tokenizer.decode(token_ids)

# Initialize OpenVINO runtime
core = Core()
model_ov = core.read_model(ONNX_MODEL_PATH)

# Force compilation to NPU (Requires Intel NPU drivers/hardware)
try:
    compiled_model = core.compile_model(model_ov, "NPU")
except Exception as e:
    print("WARNING: Failed to compile for NPU. Falling back to CPU. Ensure NPU drivers are installed.")
    compiled_model = core.compile_model(model_ov, "CPU")

if __name__ == "__main__":
    print("\n[ChatBot GPT (OpenVINO NPU) Ready — type 'exit' to quit]\n")

    while True:
        prompt = input("You: ")
        if prompt.strip().lower() == "exit":
            break

        input_ids = np.array([encode(prompt)], dtype=np.int64)

        # Pad or truncate to the fixed BLOCK_SIZE expected by ONNX
        if input_ids.shape[1] < BLOCK_SIZE:
            input_ids = np.pad(input_ids, ((0, 0), (0, BLOCK_SIZE - input_ids.shape[1])), constant_values=0)
        else:
            input_ids = input_ids[:, :BLOCK_SIZE]

        # Inference Step
        result = compiled_model([input_ids])
        generated = result[0][0]

        # Ensure 'generated' is a flat 1D list of integers
        if isinstance(generated, np.ndarray):
            generated = generated.tolist()
        if len(generated) > 0 and isinstance(generated[0], list):
            generated = [item for sublist in generated for item in sublist]

        response = decode(generated)
        response_with_lines = re.sub(r"\s*(User|Bot)\s*:", r"\n\1:", response)
        print("Bot:" + response_with_lines)
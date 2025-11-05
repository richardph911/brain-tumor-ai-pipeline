# run file to test if torch, gpu, onnx is working.

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())

# Simple tensor test
x = torch.rand(3, 3)
y = torch.rand(3, 3)
z = torch.mm(x, y)

print("Matrix multiplication result:\n", z)

# Optional: move to GPU if available
if torch.cuda.is_available():
    x_gpu = x.to("cuda")
    y_gpu = y.to("cuda")
    z_gpu = torch.mm(x_gpu, y_gpu)
    print("GPU matrix multiplication successful:", z_gpu.is_cuda)


# makes tiny model to test onnex
import torch, torch.nn as nn
class Tiny(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(64, 10)
    def forward(self, x): return self.fc(x)

m = Tiny().eval()
dummy = torch.randn(1, 64)
test_onnx_model_path = "model.onnx"
torch.onnx.export(
    m, dummy, test_onnx_model_path,
    input_names=["input"], output_names=["logits"],
    opset_version=17, dynamic_axes={"input": {0: "batch"}, "logits": {0:"batch"}}
)
print(f"Wrote {test_onnx_model_path}")

import onnxruntime as ort, numpy as np
print("Providers:", ort.get_available_providers())
sess = ort.InferenceSession(test_onnx_model_path, providers=["CUDAExecutionProvider","CPUExecutionProvider"])  # use any tiny ONNX you have
print("Active providers:", sess.get_providers())
print("Execution provider:", sess.get_provider_options())
x = np.random.randn(1,64).astype("float32")
try:
    sess.run(None, {"input": x})
    print("ORT CUDA inference OK")
except Exception as e:
    print("ORT error:", e)

# clean up
import os
# Check if ./onnx.model exists before attempting to delete it
if os.path.exists(test_onnx_model_path):
    os.remove(test_onnx_model_path)
    print(f"Cleaning up test: File '{test_onnx_model_path}' deleted successfully.")
else:
    print(f"File '{test_onnx_model_path}' does not exist.")
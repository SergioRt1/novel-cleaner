import torch

print(torch.cuda.is_available())  # Should return True if CUDA/ROCm is properly configured.
print(torch.cuda.get_device_name())
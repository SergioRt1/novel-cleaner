import torch

if __name__ == "__main__":
    print('GPU acceleration is: ', torch.cuda.is_available())  # Should return True if CUDA/ROCm is properly configured.
    print('GPU device is: ', torch.cuda.get_device_name())
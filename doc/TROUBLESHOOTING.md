# Troubleshooting

## PyTorch install on JetPack 6.x

**Do not** `pip install torch` from PyPI — those wheels have no CUDA support on Jetson aarch64.

**Step 1 — install from Jetson AI Lab:**
```bash
python3 -m pip install torch torchvision --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

**Step 2 — fix `ImportError: libcudss.so.0`**

torch ≥ 2.8 requires the CUDA Sparse Direct Solver library (`libcudss`), which JetPack 6.x does not install by default:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install libcudss0-cuda-12
```

The package installs the `.so` under a non-standard path, so `ldconfig` needs to be told about it:

```bash
echo "/usr/lib/aarch64-linux-gnu/libcudss/12" | sudo tee /etc/ld.so.conf.d/cudss.conf
sudo ldconfig
```

Verify everything works:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.10.0 True
```

Tested: torch 2.10.0 / torchvision 0.25.0 / JetPack 6.2 / CUDA 12.6

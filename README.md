Here is the absolute final, copy-pasteable `README.md`. It incorporates everything exactly as it runs in your Colab environment right now, assuming you have updated `build_tensors_all.py` in your repository.

---

# Flex-MoE: Cognitive Workload Classification

**Flexible Mixture-of-Experts for Arbitrary Modality Combination**
This repository implements a modified Flex-MoE architecture to classify cognitive workload using 9-channel EEG and ECG data. It serves as the baseline for comparing Sparse Mixture-of-Experts against custom physiological architectures.

**Note for Google Colab Users:** Ensure your runtime is set to **T4 GPU** (or any available GPU) before running the installation, as the FastMoE engine requires CUDA compilers.

## Setup & Installation

To initialize the environment in Google Colab, run the following block to clone the repository, install dependencies, and compile the custom C++ engine:

```bash
# 1. Clone Repository
!git clone https://github.com/divyaprakash1845/Flex-MoE-COGBCI
%cd /content/Flex-MoE-COGBCI

# 2. Install Python Dependencies
!pip install -q "numpy<2.0.0" "setuptools<70.0.0" wheel ninja scikit-learn

# 3. Compile FastMoE Engine (Required for Sparse MoE Routing)
%cd /content
!rm -rf fastmoe
!git clone https://github.com/laekov/fastmoe.git
%cd fastmoe
!CUDA_HOME=/usr/local/cuda USE_NCCL=0 python setup.py install

# 4. Return to Project Directory
%cd /content/Flex-MoE-COGBCI

```

## 1. Prepare Data

1. Place your raw subject folders (e.g., `sub-01`, `sub-02`) inside a folder named `raw_data`.
2. Ensure the `raw_data` folder is located in the root (`/content/`) directory of your Colab environment.

## 2. Run Preprocessing

Run the following block to extract the 1.5-second windows and stack them into task-specific master tensors.

*(Note: The bridge script is configured to automatically find the processed windows and save the final tensors directly into the repository's `raw_data/` folder).*

```bash
# Extract individual windows from raw MNE files
!python dataset.py

# Stack windows into task-specific master tensors
!python build_tensors_all.py

```

## 3. Training (Showdown Loop)

Run the following block to execute the baseline for all 4 cognitive tasks sequentially:

```bash
# Make sure you are in the project folder
%cd /content/Flex-MoE-COGBCI

!echo "=================================================="
!echo "🚀 TASK 1: PVT"
!echo "=================================================="
!python main.py --data cogbci --task PVT --modality ec --lr 1e-3 --num_experts 16 --num_layers_fus 1 --top_k 4 --train_epochs 50 --hidden_dim 64 --batch_size 16 

!echo "=================================================="
!echo "🚀 TASK 2: NBACK"
!echo "=================================================="
!python main.py --data cogbci --task NBACK --modality ec --lr 1e-3 --num_experts 16 --num_layers_fus 1 --top_k 4 --train_epochs 50 --hidden_dim 64 --batch_size 16 

!echo "=================================================="
!echo "🚀 TASK 3: FLANKER"
!echo "=================================================="
!python main.py --data cogbci --task FLANKER --modality ec --lr 1e-3 --num_experts 16 --num_layers_fus 1 --top_k 4 --train_epochs 50 --hidden_dim 64 --batch_size 16 

!echo "=================================================="
!echo "🚀 TASK 4: MATB"
!echo "=================================================="
!python main.py --data cogbci --task MATB --modality ec --lr 1e-3 --num_experts 16 --num_layers_fus 1 --top_k 4 --train_epochs 50 --hidden_dim 64 --batch_size 16 

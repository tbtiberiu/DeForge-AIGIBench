---
title: DeForge AI
emoji: 📉
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: app.py
pinned: false
---

# DeForge-AIGIBench

DeForge-AIGIBench is a comprehensive evaluation platform and benchmark for AI-Generated Image (AIGI) detection. This repository is a **modified version** of the official [AIGIBench](https://github.com/HorizonTEL/AIGIBench) (NeurIPS 2025 Datasets and Benchmarks track).

In addition to the original 15+ state-of-the-art baselines, this version integrates:
1. **DeForge-AI**: Our proposed dual-branch forensic model combining high-level vision transformer semantic representations with high-frequency noise pattern residuals.
2. **C2P-DINOv2**: An intermediary representation solution leveraging DINOv2 backbone features with common category prompting.
3. **Interactive UI**: A local and Hugging Face Spaces Gradio interface to instantly run inference using any of the 17 supported detectors.
4. **Unified Evaluation Script**: A multi-dataset benchmarking script supporting AIGC-Detection-Benchmark, MS-COCOAI, and 140k-Real-and-Fake-Faces.

---

## 🚀 Key Features

* **Gradio GUI**: Easy-to-use interface to upload any image and select from 17 pre-trained models.
* **Unified API**: All detectors are wrapped inside a consistent wrapper class implementing `.transform(img)` and `.detect(img)`.
* **Automatic Weights Downloader**: On run/import, the benchmark automatically downloads required model checkpoints from Hugging Face (`TheKernel01/DeForge-AIGIBench-Models`).
* **Diverse Generator Coverage**: Evaluates models across GANs (ProGAN, StyleGAN, BigGAN), Diffusion models (SD-v1.4, SD-v1.5, SD-XL, SD-3, FLUX, DALL-E 2/3), personalized generators (InstantID), and social media sources.

---

## 📦 Installation & Setup

The project uses `uv` for lightning-fast package management and dependency resolution.

```bash
# Clone the repository
git clone https://github.com/tbtiberiu/DeForge-AIGIBench.git
cd DeForge-AIGIBench

# Synchronize the virtual environment
uv sync
```

### Hugging Face Credentials
To download datasets and model checkpoints from the Hugging Face Hub, create a `.env` file in the root folder with your Hugging Face Access Token:

```env
HF_TOKEN=your_huggingface_token_here
```

---

## 🖥️ Running the Gradio App Locally

Launch the interactive web application to perform AIGI detection on your own images:

```bash
uv run python app.py
```

Open `http://127.0.0.1:7860` in your browser. The application will cache loaded models so switching between them does not trigger redundant reloads.

---

## 📊 Evaluating Detectors

Use `evaluate_detectors.py` to evaluate any supported model on the test splits of various Hugging Face benchmark datasets.

```bash
# General syntax
uv run python evaluate_detectors.py --model [MODEL_NAME] --dataset [DATASET_NAME] --limit [LIMIT]

# Example: Evaluate DeForge-AI on AIGC-Detection-Benchmark
uv run python evaluate_detectors.py --model DeForge-AI --dataset AIGC-Detection-Benchmark --limit 1000
```

### CLI Arguments:
* `--model`: (Required) Choose from: `AIDE`, `C2P-CLIP`, `C2P-CLIP-Original`, `C2P-DINOv2`, `CLIPDetection`, `CNNDetection`, `DeForge-AI`, `DFFreq`, `Effort`, `FreqNet`, `GramNet`, `LaDeDa`, `LGrad`, `NPR`, `RIGID`, `Resnet50`, `SAFE`.
* `--dataset`: Dataset to evaluate on: `AIGC-Detection-Benchmark` (default), `MS-COCOAI`, or `140k-Real-and-Fake-Faces`.
* `--limit`: Max images per subset split to evaluate (default: `1000`).
* `--batch_size`: Batch size for evaluation (default: `16`).
* `--num_workers`: Dataloader workers (default: `4`).
* `--show_legend`: Print descriptions for output metrics (default: `False`).

---

## 📂 Project Structure

* `app.py`: Gradio web interface for interactive, model-cached inference.
* `evaluate_detectors.py`: Robust benchmarking utility computing AUC, AP, FPR95, and Accuracy metrics.
* `detector_codes/`: Wrappers and implementations for all supported architectures.
  * `__init__.py`: Handles auto-download of weights and wraps detectors under a common class signature.
* `DeForge-AIGIBench-Models/`: Local directory housing model checkpoint files (automatically populated).
* `pyproject.toml` / `requirements.txt`: Package dependency definitions.

---

## 🔍 Supported Detectors & Checkpoints

All model checkpoints are hosted on the Hugging Face Model Hub under [TheKernel01/DeForge-AIGIBench-Models](https://huggingface.co/TheKernel01/DeForge-AIGIBench-Models) and are automatically downloaded when needed.

| Model Identifier | Reference / Publication | Venue / Year |
| :--- | :--- | :--- |
| **DeForge-AI** | *Dual-Branch Semantic + Forensic Noise Framework* | Proposed |
| **C2P-DINOv2** | *Category-Common Prompts with DINOv2* | Proposed (Intermediary) |
| **AIDE** | A Sanity Check for AI-generated Image Detection | ICLR 2025 |
| **C2P-CLIP** | C2P-CLIP: Category Common Prompt in CLIP | AAAI 2025 |
| **CLIPDetection** | Towards Universal Fake Image Detectors | CVPR 2023 |
| **CNNDetection** | CNN-generated images are surprisingly easy to spot | CVPR 2020 |
| **DFFreq** | Dual Frequency Branch Framework | TIFS 2026 |
| **Effort** | Orthogonal Subspace Decomposition for AIGI Detection | ICML 2025 |
| **FreqNet** | Frequency-Aware Deepfake Detection | AAAI 2024 |
| **GramNet** | Global Texture Enhancement for Fake Face Detection | CVPR 2020 |
| **LaDeDa** | Real-Time Deepfake Detection in the Real-World | arXiv 2024 |
| **LGrad** | Learning on Gradients: Generalized Artifacts | CVPR 2023 |
| **NPR** | Rethinking Up-Sampling Operations | CVPR 2024 |
| **RIGID** | RIGID: Robustness and Generalization in Deepfake | - |
| **Resnet50** | PyTorch Image Models (TIMM) Baseline | - |
| **SAFE** | Improving Synthetic Image Detection | KDD 2025 |

---

## 📚 Datasets

AIGIBench datasets are organized under the `TheKernel01` Hugging Face namespace:
1. **[AIGIBench Training (Setting-II)](https://huggingface.co/datasets/TheKernel01/AIGIBench)**: Balanced ProGAN and SD-v1.4 images across four categories (car, cat, chair, horse).
2. **[AIGC-Detection-Benchmark](https://huggingface.co/datasets/TheKernel01/AIGC-Detection-Benchmark)**: Evaluation subset containing test splits across 17 different generators.
3. **[MS-COCOAI](https://huggingface.co/datasets/TheKernel01/MS-COCOAI)**: Re-hosted dataset featuring SD2.1, SDXL, SD3, DALL-E 3, and Midjourney v6 images.
4. **[140k-Real-and-Fake-Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces)**: Highly balanced real vs StyleGAN face samples.

---

## 📈 Detection Results

We evaluated all 16 supported models across the test splits of the three benchmark datasets. The tables below outline the performance metrics (Accuracy, Accuracy on Real, Accuracy on Generated, AUC, AP, and FPR95) for each model.

### 1. AIGC-Detection-Benchmark

| Model         | Accuracy   | Accuracy (Real) | Accuracy (Gen) | AUC        | AP         | FPR95      |
| :------------ | :--------- | :-------------- | :------------- | :--------- | :--------- | :--------- |
| AIDE          | 0.7323     | 0.5460          | 0.9185         | 0.8400     | 0.8647     | 0.7242     |
| C2P-CLIP      | 0.9018     | 0.9460          | 0.8575         | 0.9739     | 0.9771     | 0.1470     |
| C2P-DINOv2    | 0.8255     | 0.9960          | 0.6549         | 0.9521     | 0.9518     | 0.1811     |
| CLIPDetection | 0.8406     | 0.9000          | 0.7811         | 0.9166     | 0.9261     | 0.3483     |
| CNNDetection  | 0.6595     | 0.9930          | 0.3259         | 0.8279     | 0.8199     | 0.4841     |
| DeForge-AI    | **0.9466** | 0.9930          | 0.9001         | **0.9900** | **0.9895** | **0.0391** |
| DFFreq        | 0.8364     | 0.7190          | 0.9537         | 0.9298     | 0.9404     | 0.4091     |
| Effort        | 0.8924     | 0.9470          | 0.8379         | 0.9538     | 0.9546     | 0.1691     |
| FreqNet       | 0.8203     | 0.9050          | 0.7356         | 0.8972     | 0.9109     | 0.3432     |
| GramNet       | 0.7094     | 0.9840          | 0.4347         | 0.8182     | 0.8133     | 0.4314     |
| LaDeDa        | 0.8152     | 0.9990          | 0.6314         | 0.8979     | 0.8821     | 0.2274     |
| LGrad         | 0.7262     | 0.9560          | 0.4965         | 0.8360     | 0.8178     | 0.4782     |
| NPR           | 0.7596     | 0.9940          | 0.5252         | 0.8789     | 0.8591     | 0.3294     |
| ResNet50      | 0.7500     | 0.9950          | 0.5051         | 0.8558     | 0.8465     | 0.3364     |
| RIGID         | 0.7300     | 0.6441          | 0.8159         | 0.7763     | 0.7872     | 0.6305     |
| SAFE          | 0.8089     | 0.9710          | 0.6468         | 0.9234     | 0.9176     | 0.2693     |

### 2. MS-COCOAI

| Model         | Accuracy   | Accuracy (Real) | Accuracy (Gen) | AUC        | AP         | FPR95      |
| :------------ | :--------- | :-------------- | :------------- | :--------- | :--------- | :--------- |
| AIDE          | 0.5000     | 0.9340          | 0.0660         | 0.4974     | 0.5008     | 0.9490     |
| C2P-CLIP      | 0.5018     | 0.9960          | 0.0076         | 0.5894     | 0.5876     | 0.9224     |
| C2P-DINOv2    | 0.5021     | 0.9850          | 0.0192         | 0.5392     | 0.5242     | 0.9192     |
| CLIPDetection | 0.5436     | 0.7680          | 0.3192         | 0.6232     | 0.6426     | **0.9150** |
| CNNDetection  | 0.5020     | 0.9980          | 0.0060         | 0.5617     | 0.5448     | 0.9188     |
| DeForge-AI    | **0.5893** | 0.7210          | 0.4576         | **0.6560** | **0.6593** | 0.9192     |
| DFFreq        | 0.4995     | 0.9960          | 0.0030         | 0.5021     | 0.5012     | 0.9538     |
| Effort        | 0.5001     | 0.9990          | 0.0012         | 0.5620     | 0.5563     | 0.9226     |
| FreqNet       | 0.4998     | 0.9960          | 0.0036         | 0.5109     | 0.5077     | 0.9456     |
| GramNet       | 0.5013     | 0.9990          | 0.0036         | 0.5128     | 0.5074     | 0.9404     |
| LaDeDa        | 0.5000     | 1.0000          | 0.0000         | 0.4991     | 0.4996     | 0.9948     |
| LGrad         | 0.4998     | 0.9900          | 0.0096         | 0.4997     | 0.4999     | 0.9482     |
| NPR           | 0.4995     | 0.9990          | 0.0000         | 0.4987     | 0.4993     | 0.9966     |
| ResNet50      | 0.4997     | 0.9990          | 0.0004         | 0.5084     | 0.5044     | 0.9350     |
| RIGID         | 0.5208     | 0.6092          | 0.4324         | 0.4959     | 0.4962     | 0.9536     |
| SAFE          | 0.5016     | 0.9940          | 0.0092         | 0.5008     | 0.4949     | 0.9436     |

### 3. 140k-Real-and-Fake-Faces

| Model         | Accuracy   | Accuracy (Real) | Accuracy (Gen) | AUC        | AP         | FPR95      |
| :------------ | :--------- | :-------------- | :------------- | :--------- | :--------- | :--------- |
| AIDE          | 0.5015     | 0.9970          | 0.0060         | 0.5310     | 0.5391     | 0.9320     |
| C2P-CLIP      | 0.5025     | 0.9990          | 0.0060         | 0.6923     | 0.6914     | 0.8130     |
| C2P-DINOv2    | 0.5160     | 1.0000          | 0.0320         | 0.8795     | 0.8568     | 0.4440     |
| CLIPDetection | 0.7375     | 0.6460          | 0.8290         | 0.8306     | 0.8342     | 0.6210     |
| CNNDetection  | 0.4985     | 0.9960          | 0.0010         | 0.5930     | 0.5594     | 0.9120     |
| DeForge-AI    | **0.9080** | 0.9110          | 0.9050         | **0.9676** | **0.9696** | 0.1720     |
| DFFreq        | 0.5000     | 1.0000          | 0.0000         | 0.5030     | 0.5015     | 0.9860     |
| Effort        | 0.6715     | 1.0000          | 0.3430         | 0.9625     | 0.9514     | **0.1190** |
| FreqNet       | 0.4995     | 0.9710          | 0.0280         | 0.5011     | 0.5110     | 0.9480     |
| GramNet       | 0.5000     | 1.0000          | 0.0000         | 0.4907     | 0.4951     | 0.9580     |
| LaDeDa        | 0.5000     | 1.0000          | 0.0000         | 0.5000     | 0.5000     | 1.0000     |
| LGrad         | 0.4760     | 0.7860          | 0.1660         | 0.4721     | 0.4851     | 0.9590     |
| NPR           | 0.5000     | 1.0000          | 0.0000         | 0.5000     | 0.5000     | 1.0000     |
| ResNet50      | 0.5000     | 1.0000          | 0.0000         | 0.4895     | 0.4947     | 0.9750     |
| RIGID         | 0.8480     | 0.8270          | 0.8690         | 0.9159     | 0.9205     | 0.4110     |
| SAFE          | 0.5000     | 1.0000          | 0.0000         | 0.5376     | 0.5240     | 0.9150     |

---

## 📝 Citation
If you find this benchmark or the models helpful in your research, please cite the original AIGIBench work:

```bibtex
@inproceedings{li2025artificial,
  title={Is Artificial Intelligence Generated Image Detection a Solved Problem?},
  author={Li, Ziqiang and Yan, Jiazhen and He, Ziwen and Zeng, Kai and Jiang, Weiwei and Xiong, Lizhi and Fu, Zhangjie},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

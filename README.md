# Cross-scale Content Adaptive Network for 3D Multi-Object Tracking and Fish Activity Quantification

## This repository contains the official implementation of the paper "Cross-scale Content Adaptive Network for Three-dimensional Multi-Object Tracking and Fish Activity Quantification" <br />

**The model code for generating 2D trajectories is in the model/ directory** <br />
**The evaluation code and results are in the eval_result/ directory** <br />
**Ground truth annotations are provided in the annotations/ directory** <br />

## 📁 Repository Structure

```
.
├── model/                 # Training code for generating 2D trajectories
├── eval_result/          # Evaluation scripts and trajectory results
│   ├── ZebraFish_05/     # Camera parameters and results for video 05
│   ├── ZebraFish_06/     # Camera parameters and results for video 06
│   ├── ZebraFish_07/     # Camera parameters and results for video 07
│   └── ZebraFish_08/     # Camera parameters and results for video 08
└── annotations/          # Ground truth trajectory annotations
```

## 🚀 Features

- 2D multi-object tracking using a cross-scale content adaptive network
- 3D trajectory reconstruction from multi-view 2D trajectories
- Zebrafish activity quantification and behavioral analysis
- Evaluation scripts for comparing predicted and ground truth trajectories

## 🛠️ Installation

### Prerequisites

- Python 3.6+
- PyTorch 1.7+
- CUDA 11.0+ (for GPU acceleration)

### Install Dependencies

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
pip install -r requirements.txt
```

## 📊 Usage

### 1. Generate 2D Trajectories

Run the model to generate 2D trajectories:

```bash
cd model
python train.py  # for training
python infer.py  # for inference
```

The output 2D trajectories will be saved as TXT files named in the format `fish3d_front_joint_cgaivb.txt`.

### 2. Evaluate 2D and 3D Trajectories

Move the generated trajectory files to the `eval_result` directory under the corresponding subfolder (e.g., `ZebraFish_05`). Then run:

```bash
python evaluate.py --video ZebraFish_05
```

This will generate:
- `tracklets_2d_*.csv`: 2D trajectories for each view
- `tracklets_3d.csv`: Reconstructed 3D trajectories

### 3. Quantitative Evaluation

To compare against ground truth annotations:

```bash
python MOT_evaluation.py
```

## 📝 File Formats

**Generated 2D trajectories**: TXT files with naming convention `fish3d_front_joint_cgaivb.txt` <br />
**Output 2D trajectories**: CSV files with naming convention `tracklets_2d_*.csv` <br />
**Output 3D trajectories**: CSV file named `tracklets_3d.csv` <br />

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{yourpaper,
  title={Cross-scale Content Adaptive Network for 3D Multi-Object Tracking and Fish Activity Quantification},
  author={Author1, Author2, ...},
  journal={Journal Name},
  year={2023}
}
```

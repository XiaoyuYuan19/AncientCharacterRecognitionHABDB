# Ancient Character Recognition with Multi-Model Ensemble

[![Demo](https://img.shields.io/badge/Live-Demo-blue)](https://houmaocr.ddns.net/)
[![Video](https://img.shields.io/badge/YouTube-Demo-red)](https://www.youtube.com/watch?v=myvIOghtj1k)
[![License](https://img.shields.io/badge/License-See%20Notice-orange)](#license-and-copyright)

> **A deep learning system for recognizing ancient Chinese characters from Houma Alliance Book artifacts**  
> Bachelor Thesis Research | Nanjing Institute of Technology & University of Oulu  
> June 2021 – June 2023

[Live Demo](https://houmaocr.ddns.net/) • [Watch Video](https://www.youtube.com/watch?v=myvIOghtj1k) • [Report Issue](#)

---

## 🏆 Achievements

- **🥇 1st Prize** - Chinese Collegiate Computing Competition (National-Level, 2023)
- **🥈 2nd Prize** - Jiangsu Province Bachelor Thesis Award (Provincial-Level, 2023)
- **📜 Patent Granted** - CN202311008755.1 (Published)
- **💰 Funded Research** - National College Student Innovation Program (¥20,000 CNY)

---

## 📖 Overview

This project addresses the challenge of digitizing and recognizing ancient Chinese characters from historical artifacts, specifically the Houma Alliance Book (侯马盟书) - important Spring and Autumn period archaeological relics. We built the **first digital database** for these characters and developed a multi-model deep learning system achieving **94.2% accuracy**.

### Key Features

- 🗂️ **First digital database** of Houma Alliance Book characters (297 classes, 3,547 samples)
- 🤖 **Multi-model ensemble** combining ResNet-18, Vision Transformer, and Cross-attention Transformer
- 📊 **Long-tail optimization** using Mixup augmentation for imbalanced datasets
- 🌐 **Full-stack OCR system** with real-time detection and recognition
- 🎨 **Interactive web interface** for researchers and historians

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  Frontend (JS)  │
│  Upload & View  │
└────────┬────────┘
         │
┌────────▼────────┐
│ Flask Backend   │
│ Image Processing│
└────────┬────────┘
         │
    ┌────┴────┐
    │ YOLOv5  │ Detection
    └────┬────┘
         │
┌────────▼──────────┐
│ Ensemble Models   │
├───────────────────┤
│ • CNN (Custom)    │
│ • ResNet-18       │
│ • LeNet           │
│ • AlexNet         │
└────────┬──────────┘
         │
    Recognition Results
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA 11.0+ (for GPU support)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/houma-ocr.git
cd houma-ocr
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download model weights**
```bash
# Model weights are not included in the repository
# Contact the authors for access to pre-trained models
```

4. **Run the application**
```bash
python app.py
```

The application will be available at `http://localhost:5000`


---

## 💡 Technical Highlights

### 1. Multi-Model Ensemble Framework

We developed a decision-level fusion approach combining four deep learning architectures:

- **Custom CNN**: Baseline architecture optimized for ancient character features
- **ResNet-18**: Deep residual learning for robust feature extraction
- **LeNet**: Lightweight model for comparison
- **AlexNet**: Classic architecture for ensemble diversity

**Ensemble Strategy**: Linear weight optimization at decision level achieves 94.2% accuracy on test set.

### 2. Long-Tail Data Handling

**Challenge**: Head 20% of classes contain 60% of samples (severe imbalance)

**Solution**: Mixup-based data augmentation targeting tail classes
- Generates synthetic samples through linear interpolation
- Improves tail-class recognition by **12%** over single-model baselines
- Maintains head-class performance while boosting rare character recognition

### 3. Detection Pipeline

- **YOLOv5** for character localization in document images
- Custom preprocessing for ancient manuscript characteristics
- Bounding box refinement for tightly-packed characters

---

## 📊 Dataset

We established the **first digital database** for Houma Alliance Book characters:

- **297 character classes** (ancient Chinese script variants)
- **3,547 total samples** from:
  - Museum artifact photographs
  - Expert human imitations
  - Data augmentation techniques

> ⚠️ **Note**: The dataset is **not publicly available** due to copyright restrictions and cultural heritage protection regulations. For research collaboration or data access requests, please contact the corresponding author.

---

## 🖥️ Usage

### Web Interface

1. **Upload Image**: Drag and drop or select ancient character images
2. **Detection**: System automatically detects character regions
3. **Recognition**: Multi-model ensemble predicts each character
4. **Visualization**: View results with confidence scores

### API Usage (if applicable)

```python
from app import get_model, predict

# Load model
model = get_model()

# Predict single image
result = predict(model, "path/to/image.png")
print(f"Recognized character: {result}")
```

---

## 📄 Publications

### Conference Papers

1. **X. Yuan**, Z. Zhang, Y. Sun, Z. Xue, X. Shao, & X. Huang. (2023).  
   *A new database of Houma Alliance Book ancient handwritten characters and its baseline algorithm.*  
   Proc. of the 8th Int. Conf. on Multimedia Systems and Signal Processing (ICMSSP '23). ACM.  
   DOI: [10.1145/3613917.3613923](https://doi.org/10.1145/3613917.3613923)

2. Z. Zhang, X. Huang, **X. Yuan**, & Y. Sun. (2023).  
   *HABFD: Houma Alliance Book facsimiles database.*  
   Proc. of the IEEE Int. Conf. on Image, Vision and Computing (ICIVC '23). IEEE.  
   DOI: [10.1109/icivc58118.2023.10269984](https://doi.org/10.1109/icivc58118.2023.10269984)

### Patent

**X. Yuan (First Inventor)**, Z. Zhang, X. Huang. (2023).  
*Method for ancient Chinese character sample collection, detection and recognition based on optimized deep network.*  
Chinese Patent **CN202311008755.1** (Published)

### Software Copyright

**X. Yuan (First Copyright Holder)**, Z. Zhang, X. Huang, et al. (2022).  
*Houma Alliance Book Intelligent Recognition Platform v1.0.*  
Software Copyright Registration No. **2022SR1030217**, National Copyright Administration of China

---

## 🛠️ Project Structure

```
houma-ocr/
├── app.py                 # Flask application entry point
├── server.py             # Tornado production server
├── detect.py             # YOLOv5 detection module
├── trainModelTest.py     # Model training utilities
├── predictInterface.py   # Prediction interface for ensemble
├── json2png.py          # Data preprocessing scripts
├── requirements.txt      # Python dependencies
├── templates/           # HTML templates
│   ├── cover.html
│   ├── index.html
│   ├── predict.html
│   └── detection.html
├── static/              # Static assets
│   ├── images/
│   ├── dataset0729/
│   └── download/
├── weights/             # Model weights (not included)
├── npy/                # Saved model checkpoints
│   ├── Res/
│   ├── Alex/
│   └── Le/
└── model/              # Model architectures
    ├── alexnet.py
    ├── lenet.py
    └── resnet.py
```

---

## 🤝 Contributing

This is a research project with specific copyright and intellectual property considerations. If you're interested in:

- **Research collaboration**
- **Dataset access**
- **Model improvements**
- **Commercial applications**

Please contact the project lead or supervisor before making contributions.

---

## 👥 Team

**Project Lead & First Author**  
**Xiaoyu Yuan** (袁筱钰)  
Nanjing Institute of Technology  
Email: [your-email@example.com]

**Supervisor**  
**Prof. Huang**
Nanjing Institute of Technology

**Collaborators**  
Z. Zhang, Y. Sun, Z. Xue, X. Shao  

---

## 📜 License and Copyright

### Important Copyright Notice

⚠️ **This project contains protected intellectual property:**

1. **Patent Protection**: The core recognition method is protected by Chinese Patent CN202311008755.1
2. **Software Copyright**: Platform registered as Software Copyright No. 2022SR1030217
3. **Dataset Rights**: The Houma Alliance Book character database is NOT publicly available due to:
   - Cultural heritage protection regulations
   - Museum artifact copyright restrictions
   - Academic research collaboration agreements

### Usage Restrictions

- **Academic Research**: Contact authors for collaboration opportunities
- **Commercial Use**: Requires explicit written permission and licensing
- **Dataset Access**: Available only through approved research partnerships

### Code License

The code implementation (excluding models and data) is available for **non-commercial academic use only**. For any other use cases, please contact the authors.


---

## 🙏 Acknowledgments

- **Funding**: National College Student Innovation Program (¥20,000 CNY)
- **Institutions**: Nanjing Institute of Technology, University of Oulu
- **Museums**: Houma Museum for artifact access and collaboration
- **Experts**: Calligraphy experts for character annotation and validation

---

## 📚 Citation

If you reference this work in your research, please cite:

```bibtex
@inproceedings{yuan2023houma,
  title={A new database of Houma Alliance Book ancient handwritten characters and its baseline algorithm},
  author={Yuan, Xiaoyu and Zhang, Z. and Sun, Y. and Xue, Z. and Shao, X. and Huang, X.},
  booktitle={Proceedings of the 8th International Conference on Multimedia Systems and Signal Processing},
  pages={},
  year={2023},
  organization={ACM},
  doi={10.1145/3613917.3613923}
}
```

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for ancient Chinese cultural heritage preservation

</div>

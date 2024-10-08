# ML Processor: Detect Non-Content Sentences in Novels

### Overview

`ml_processor` is a Python package designed to detect non-content sentences in novels using machine learning. It helps identify sentences that are likely filler, redundant, or otherwise non-contributive to the plot.

### Features

- **Machine Learning Based Classification**: Detect non-content sentences using pre-trained ML models.
- **Text Preprocessing**: Automatically preprocesses the text by normalizing formatting and removing irrelevant characters.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SergioRt1/novel-cleaner.git
   cd ml_processor
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

3. Ensure Python 3.8 or later is installed.

4. (Optional) If using PyTorch with ROCm:
   ```bash
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.2
   ```

### Dependencies

The required dependencies will be automatically installed with the package. You can also manually install them:

```bash
pip install -r requirements.txt
```

Main dependencies:
- `scikit-learn`
- `datasets`
- `torch`
- `transformers`
- `tkinter` (requires separate installation)

### Usage

You can use the package to detect non-content sentences once it's installed. Example usage:

```bash
python -m ml_processor.prediction
```

### Training the Model

To train the model on your own data, prepare a labeled dataset with `content` and `non-content` sentences and run the following script:

```bash
python train_model.py
```

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

### Folder Structure

```bash
ML/
├── ml_processor/
│   ├── __init__.py
│   ├── prediction.py
│   ├── train_model.py
│   ├── gpu_validation.py
│   ├── labler.py
│   ├── manual_labler.py
│   └──split.py
├── ml_data/
│   ├── non-novel.txt
│   └── novel-like.txt
├── README.md
├── requirements.txt
├── setup.py
└── LICENSE
```

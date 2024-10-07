# ML Processor: Detect Non-Content Sentences in Novels

### Overview

`ml_processor` is a Python package designed to detect non-content sentences in novels using machine learning techniques. This project aims to help identify sentences that are likely non-contributive to the plot or content, such as filler lines, promotional phrases, redundant phrases, or meta-commentary.

### Features

- **Machine Learning Based Sentence Classification**: Detect non-content sentences based on trained ML models.
- **Text Preprocessing**: Automatically preprocesses the novel text, removing irrelevant characters, normalizing formatting, etc.

### Installation

You can install the package by cloning the repository and running the following command:

```bash
git clone https://github.com/SergioRt1/novel-cleaner.git
cd ml_processor
pip install -e .
```

Ensure you have Python 3.8 or later installed.

### Dependencies

The following dependencies are required for `ml_processor` to work. These will be automatically installed when running the `pip install -e .` command:

- `scikit-learn`
- `datasets`
- `torch`
- `transformers`
- `tkinter` This one requires a separated instalation
  
You can also install them manually by running:

```bash
pip install -r requirements.txt
```

### Usage


### Training the Model 

If you want to train the model on your own dataset, the training scripts are included in the repository. You can prepare a labeled dataset where each sentence is marked as `content` or `non-content` and run the training script.

```bash
python train_model.py
```


#### Basic Example


Once the package is installed, you can easily use it to detect non-content sentences in a novel. Here's a simple example:


### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


---

### Sample Folder Structure

```bash
ML/
├── ml_processor/
│   ├── __init__.py
│   ├── prediction.py
│   ├── train_model.py
├── data/
│   └── training_data.csv
├── utils/
│   ├── gpu_validation.py
│   ├── labler.py
├── README.md
├── requirements.txt
├── setup.py
└── LICENSE
```


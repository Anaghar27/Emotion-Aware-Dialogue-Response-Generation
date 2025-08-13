# Emotion-Aware-Dialogue-Response-Generation

## NLP Project – Environment Setup Guide

This guide outlines the steps to set up the development environment for the **NLP Project**.

---

### **1. Check Existing Environments**

Before creating a new environment, check your existing Conda environments and Jupyter kernels:

```bash
conda env list
jupyter kernelspec list
```

---

### **2. Create and Activate the Environment**

Create a new Conda environment named `nlp_project` with Python 3.9 and activate it:

```bash
conda create -n nlp_project python=3.9 -y
conda activate nlp_project
```

---

### **3. Install IPython Kernel**

Install the IPython kernel for the environment:

```bash
conda install ipykernel -y
```

---

### **4. Register the Kernel with Jupyter**

Register the new environment as a Jupyter kernel:

```bash
python -m ipykernel install \
    --user \
    --name nlp_project \
    --display-name "Python (nlp_project)"
```

---

### **5. Create `requirements.txt`**

Create a `requirements.txt` file with the following dependencies:

```txt
# Core ML & NLP
torch>=1.13.1
transformers>=4.30.0

# Data handling & preprocessing
numpy>=1.24.3
pandas>=2.0.2
nltk>=3.8.1
datasets>=2.13.1
kaggle>=1.5.17

# Modeling utilities
scikit-learn>=1.2.2
tqdm>=4.65.0

# Visualization & EDA
matplotlib>=3.7.1
seaborn>=0.12.2
```

---

### **6. Install Project Dependencies**

Run the following command to install all dependencies:

```bash
pip install -r requirements.txt
```

---

## **Notes**

- Ensure you are inside the `nlp_project` environment before installing packages.
- For GPU support with PyTorch, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

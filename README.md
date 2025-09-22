#  Machine Learning Assignments – From Scratch

This repository contains my implementations of **three major machine learning assignments** completed during my coursework.  
Each assignment focuses on a core area of machine learning: **regression**, **classification**, **feature selection**, and **model evaluation** — all built from scratch using Python.

---

##  Assignments Overview

### Assignment 1 – Regression Models from Scratch

Polynomial regression using both **Normal Equation** and **Gradient Descent**.

- Linear regression from scratch  
- Polynomial feature expansion  
- MSE cost function  
- Comparison of optimization techniques  
- Visualized model fit

**Modules used**:  
`MachineLearningModel.py`

---

### Assignment 2 – Binary Classification & Decision Boundaries

Logistic regression with both linear and nonlinear decision boundaries.

- Binary classification using logistic regression  
- Feature expansion using polynomial mapping  
- Sigmoid + cost function + gradient descent  
- Visualization of decision boundaries

**Modules used**:  
`MachineLearningModel.py`, `DecisionBoundary.py`  
**Datasets**: `fashion_subset.csv`, `iris_data.csv`, `wine_combined.csv`

---

### Assignment 3 – Feature Selection, ROC, and Real Datasets

Training and evaluating models on real-world datasets with feature selection and ROC analysis.

- Forward feature selection (greedy method)  
- Custom F-score, precision, recall, true/false positive rate  
- Training and evaluation using real datasets

**Modules used**:  
`ForwardSelection.py`, `ROCAnalysis.py`, `MachineLearningModel.py`  
**Datasets**: `bank-additional-full.csv`, `heart.csv`

---

## 🛠️ Technologies Used

- Python 3.x  
- NumPy  
- Pandas  
- Matplotlib  
- Jupyter Notebook

>  No machine learning libraries (`scikit-learn`, etc.) were used for model training. All algorithms were implemented manually.

---

##  How to Run

Follow these steps to set up and run the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**On macOS/Linux:**

```bash
source venv/bin/activate
```

**On Windows:**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` file yet, you can create one using:

```bash
pip freeze > requirements.txt
```

### 4. Run the Jupyter notebooks

```bash
jupyter notebook
```

Then open any notebook (e.g., `mh224tb_A2.ipynb` or `mh224tb_A3.ipynb`) and run the cells.

---

##  Notes

- The Bank Marketing dataset used is from [Moro et al., 2014].  
  DOI: [10.1016/j.dss.2014.03.001](http://dx.doi.org/10.1016/j.dss.2014.03.001)

- This repository is part of my machine learning studies and serves as a personal learning portfolio.

---

## 📧 Contact

**Mustafa Habeb**  
GitHub: [@mustafaqh](https://github.com/yourusername)  
Email: mustafa.habeb93@gmail.com

# Implementation and Application of Classification Methods

This repository contains the implementation and analysis of classification methods for the course **"Foundations of Data Mining"** at **Brandenburgische Technische Universität Cottbus-Senftenberg**. The repository includes two practical training tasks (PR1 and PR2) that focus on clustering and classification methods.

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Running the Experiment](#running-the-experiment)
- [Project Structure](#project-structure)
- [Results](#results)
- [Practical Training Tasks](#practical-training-tasks)
<!-- - [License](#license) -->

## Setup Instructions

To set up the environment and run the experiments, follow these steps:

### Prerequisites

1. **Python**: Ensure Python 3.8 or above is installed on your system. You can download it from [python.org](https://www.python.org/).
2. **Git**: Clone the repository using Git or download it as a ZIP file.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fharookshaik/BTU_FDM.git
   cd BTU_FDM
   ```

2. (Recommended) Set up a Conda environment:
   - Ensure Conda is installed. If not, download and install it from [Conda's website](https://docs.conda.io/en/latest/).
   - Create a new Conda environment using the provided `requirements.yml` file:
     ```bash
     conda env create -f requirements.yml
     conda activate btu_fdm_env
     ```

3. (Optional) Alternatively, set up a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

4. Install the required Python libraries (if using a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

   Note: The `requirements.yml` file is recommended for an easier setup using Conda.

## Running the Experiment

### Practical Task 1 (PR1): Clustering with KMeans and DBSCAN

1. Navigate to the `PR1` directory:
   ```bash
   cd PR1
   ```

2. Open and execute the Jupyter notebook `image_clustering.ipynb`:
   ```bash
   jupyter notebook image_clustering.ipynb
   ```

   This notebook:
   - Imports and utilizes the custom clustering algorithms (KMeans and DBSCAN) implemented in `algorithms.py`.
   - Demonstrates clustering on image data such as `giraffe.png`.
   - Outputs results such as `dbscan_output.jpg` and `kmeans_output.jpg`.

### Practical Task 2 (PR2): Classification with Custom KNN

1. Navigate to the `PR2` directory:
   ```bash
   cd PR2
   ```

2. Open and execute the Jupyter notebook `pr2_clustering.ipynb`:
   ```bash
   jupyter notebook pr2_clustering.ipynb
   ```

   This notebook:
   - Imports and utilizes the custom KNN classifier implemented in `algorithms.py`.
   - Demonstrates classification on datasets such as `g008_d2_c1.csv`, `g008_d2_c2.csv`, etc.

## Project Structure

```
BTU_FDM/
├── PR1
│   ├── FDM_PR01.pdf               # PR1 Task Description
│   ├── algorithms.py              # Custom KMeans and DBSCAN implementations
│   ├── dbscan_output.jpg          # DBSCAN clustering output
│   ├── giraffe.png                # Input image for clustering
│   ├── image_clustering.ipynb     # Jupyter notebook for clustering models
│   └── kmeans_output.jpg          # KMeans clustering output
├── PR2
│   ├── FDM_PR02.pdf               # PR2 Task Description
│   ├── algorithms.py              # Custom KNN implementation
│   ├── g008_d2_c1.csv             # Confusion Matrix for Breast Cancer Dataset run on CustomKNNClassifier
│   ├── g008_d2_c2.csv             # Confusion Matrix for Breast Cancer Dataset run on KNeighborsClassifier
│   ├── g008_d2_c3.csv             # Confusion Matrix for Breast Cancer Dataset run on Gaussian NaiveBayes
│   ├── g008_d3_c1.csv             # Confusion Matrix for Wine Quality Dataset run on CustomKNNClassifier
│   ├── g008_d3_c2.csv             # Confusion Matrix for Wine Quality Dataset run on KNeighborsClassifier
│   ├── g008_d3_c3.csv             # Confusion Matrix for Wine Quality Dataset run on Gaussian NaiveBayes
│   └── pr2_clustering.ipynb       # Jupyter notebook for classification models
├── PR2_Report.docx                # Word version of PR2 report
├── PR2_Report.pdf                 # PDF version of PR2 report
├── README.md                      # Project documentation (this file)
├── requirements.txt               # Required Python packages
├── requirements.yml               # Conda environment setup file
```

## Results

### Summary of Results

1. **PR1 (Clustering)**:
   - Demonstrated clustering on image data using KMeans and DBSCAN.
   - Outputs: `dbscan_output.jpg`, `kmeans_output.jpg`.

2. **PR2 (Classification)**:
   - Confusion matrices for each dataset ran on respective classifier is provided in CSV files.
   - Outputs: Classification accuracy, confusion matrices, and runtime metrics.

## Practical Training Tasks

### PR1: Clustering Task

- **Algorithms**: Custom implementations of KMeans and DBSCAN.
- **Objective**: Cluster image data and visualize results.
- **Key Files**:
  - `algorithms.py`: Contains clustering algorithm implementations.
  - `image_clustering.ipynb`: Jupyter notebook for clustering.

### PR2: Classification Task

- **Algorithm**: Custom implementation of KNN Classifier.
- **Objective**: Classify datasets using the custom KNN implementation.
- **Key Files**:
  - `algorithms.py`: Contains KNN implementation.
  - `pr2_clustering.ipynb`: Jupyter notebook for classification.

<!-- ## License

This project is licensed under the MIT License. See the LICENSE file for details. -->

---

For any questions or issues, feel free to open an issue in the repository.

README.txt

Installation
-------------
1. Set up the environment:
   - Recommended: Create a Conda environment using the `requirements.yml` file:
     ```
     conda env create -f requirements.yml
     conda activate fdm
     ```
   - Alternatively, set up a virtual environment and install dependencies:
     ```
     python -m venv env
     source env/bin/activate  # On Windows: env\Scripts\activate
     pip install -r requirements.txt
     ```


Execution
---------
PR1 (Clustering)
1. Navigate to the `PR1` directory:
   ```
   cd PR1
   ```
2. Open and run the Jupyter notebook `image_clustering.ipynb`:
   ```
   jupyter notebook image_clustering.ipynb
   ```
3. This notebook performs clustering using custom KMeans and DBSCAN algorithms on `giraffe.png` and visualizes the outputs as `dbscan_output.jpg` and `kmeans_output.jpg`.

PR2 (Classification)
1. Navigate to the `PR2` directory:
   ```
   cd PR2
   ```
2. Open and run the Jupyter notebook `pr2_clustering.ipynb`:
   ```
   jupyter notebook pr2_clustering.ipynb
   ```
3. This notebook demonstrates classification using the custom KNN implementation on datasets such as `g008_d2_c1.csv` and outputs accuracy, confusion matrices, and runtime metrics.


Operation
---------
- Custom Algorithms: Implementations for clustering (PR1) and classification (PR2) are located in `algorithms.py` within their respective folders.
- Results: The notebooks display results interactively. Outputs include visualizations (e.g., clustering plots) and evaluation metrics (e.g., accuracy, confusion matrix etc.).
- Reports: Detailed analysis for Practice Training - Task 2 can be found in `PR2_Report.pdf`. 


Project Structure:
-----------------
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
├── README.md                      # Project documentation
├── README.txt                     # Short Project documentation (txt)
├── requirements.txt               # Required Python packages
├── requirements.yml               # Conda environment setup file

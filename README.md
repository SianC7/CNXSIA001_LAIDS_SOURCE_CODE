# CNXSIA001_LAIDS_SOURCE_CODE
# LAIDS: Lightweight AI-based Intrusion Detection Systems for Resource-Constrained Networks

# Project Overview
Low-resource networks, such as Internet of Things networks, face continuously evolving and sophisticated threats. Traditional Intrusion Detection Systems (IDSs) have become inefficient, and AI-based approaches are often impractical due to their high computational demands. Thus there is a critical gap: the absence of lightweight AI-based IDSs that deliver high detection accuracy while remaining feasible for deployment in resource-constrained networks. This project aims to address this gap by designing and evaluating lightweight AI-IDS models that balance strong detection capabilities with low resource usage. Three core model architectures are explored: a baseline 1D-CNN, a PCA-CNN and an AE-MLP, which are trained and evaluated with the CICIDS2017 dataset. These models, along with their quantised variants, are evaluated on their classification performance, resource efficiency and their ability to generalise to ’zero-day attacks’.

The investigation is guided by these research questions:
1. Can a PCA-CNN or AE-MLP IDS achieve higher resource efficiency than a baseline 1D-CNN IDS, while maintaining comparable or superior performance in a resource-constrained environment?
2. Can a quantised version of a PCA-CNN or AE-MLP IDS achieve higher resource efficiency than a quantised version of the 1D-CNN IDS baseline, while maintaining comparable or superior performance in a resource-constrained environment?
3. Can PCA-CNN or AE-MLP IDS generalise to unseen attacks more effectively than a baseline 1D-CNN?

# Usage
To run the experimental code without regenerating the datasets:
1. The full project folder containing the datasets and models can be found at https://drive.google.com/drive/folders/1xw92oMqQRnlDOngFd13FPXQa9J8mIBgh?usp=share_link 
1. Open the Jupyter notebooks in Google Colab, which includes all the required libraries pre-installed.
2. Run all programmes in a Python 3 runtime environment using the T4 GPU. The resource usage experiment files are an exception and must be run on the Google Colab CPU to ensure accurate resource usage measurements.
3. For most programmes, you only need to update the file paths so that they point to the correct dataset and model locations, then click ‘Run all’ in Colab. Any additional instructions specific to certain programmes are included within their respective files.
4. The two main file paths that may need updating are dataset_path and save_path/model_path, which are defined at the top of each Colab notebook.
5. To change a file path in Google Colab, go to: File > Drive > MyDrive, then navigate to the CNXSIA001_LAIDS_SOURCE_CODE folder where the dataset and model files are stored.

# Experiment Pipeline
Follow the steps below to regenerate all datasets and models required to re-run the experiments from scratch:
1. Data Pre-processing:
   - Run the MODEL_DATA_PREPROCESSING.ipynb notebook using the cleaned CICIDS2017 dataset (cicids2017_cleaned.csv) available on Kaggle at https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed.
   - Ensure that the file paths in the notebook are updated to the correct file locations for saving the processed datasets.
2. FP32 Model Training:
   - Once the datasets have been generated, run the FP32 model notebooks (for example, FP32_PCA_1DCNN_Model.ipynb) to train and save the FP32 versions of each model.
   - Note: The Baseline 1D-CNN FP32 and quantised models are both generated and tested within a single notebook.
3. Quantised Model Generation & Experiment:
   - Run the Experiment 2 (quantisation) notebooks to generate the FP32, FP16, Dynamic INT8, and Full INT8 TFLite models for each TensorFlow FP32 model (for example, Experiment_2_Quant_PCA_CNN_Models.ipynb).
4. Resource Usage and Generalisation Experiments:
   - Finally, run the resource usage and generalisation experiment notebooks to measure the resource efficiency of the TensorFlow and TFLite models and assess their ability to detect unseen attacks.

# Acknowledgements
- The baseline 1D-CNN model used in the project was developed in collaboration with Claire Campbell (@clair3campb3ll) and Christopher Blignaut (@ChrispyBacon123).
- The original CICIDS2017 dataset was developed by the Canadian Institute of Cybersecurity (CIC) to support the training and evaluation of machine learning models for intrusion detection, https://www.unb.ca/cic/datasets/ids-2017.html.
- The processed version (cicids2017_cleaned.csv) used in this project was prepared by Eric Anacleto Ribeiro, https://www.kaggle.com/ericanacletoribeiro.
  
# Versions of critical software used
- Python version 3.12.11
- TensorFlow/TFLite version 2.19.0
- Scikit-learn version 1.6.1
- NumPy version 2.0.2
- Imbalanced-learn version 0.14.0
- psutil 5.9.5

# Source code table of contents:
```text
CNXSIA001_LAIDS_SOURCE_CODE/
├── MODEL_DATA_PREPROCESSING.ipynb
│   
├── Baseline 1D CNN Model Files/
│   ├── Datasets/
│   │   ├── adasyn_baseline_X_train.csv
│   │   ├── adasyn_baseline_y_train.csv
│   │   ├── baseline_X_test.csv
│   │   ├── baseline_X_val.csv
│   │   ├── baseline_y_test.csv
│   │   └── baseline_y_val.csv
│   │
│   ├── Models/
│   │   ├── Best_Baseline_float32.tflite
│   │   ├── Best_Baseline_fp16_weights.tflite
│   │   ├── Best_Baseline_int8_full.tflite
│   │   ├── Best_Baseline_int8_weights.tflite
│   │   └── Best_Baseline.keras
│   │
│   ├── Baseline_1DCNN_&_Experiment_2_Quant_Models.ipynb
│   ├── Baseline_1DCNN_Experiment_1_Resource_Usage.ipynb
│   ├── Baseline_1DCNN_Experiment_3_Generalisation_Bots.ipynb
│   └── Baseline_1DCNN_Experiment_3_Generalisation_DoS.ipynb
│   
├── PCA-CNN Model Files/
│   ├── Datasets/
│   │   ├── adasyn_pca_cnn_X_train.csv
│   │   ├── adasyn_pca_cnn_y_train.csv
│   │   ├── pca_cnn_X_test.csv
│   │   ├── pca_cnn_X_val.csv
│   │   ├── pca_cnn_y_test.csv
│   │   └── pca_cnn_y_val.csv
│   │
│   ├── Models/
│   │   ├── Best_PCA_CNN_float32.tflite
│   │   ├── Best_PCA_CNN_fp16_weights.tflite
│   │   ├── Best_PCA_CNN_int8_full.tflite
│   │   ├── Best_PCA_CNN_int8_weights.tflite
│   │   ├── Best_PCA_CNN.keras
│   │   └── BEST_PCA.pkl
│   │
│   ├── FP32_PCA_1DCNN_Model.ipynb
│   ├── Experiment_2_Quant_PCA_CNN_Models.ipynb
│   ├── PCA_CNN_Experiment_1_Resource_Usage.ipynb
│   ├── PCA_CNN_Experiment_3_Generalisation_Bots.ipynb
│   └── PCA_CNN_Experiment_3_Generalisation_DoS.ipynb
│   
├── AE-MLP Model Files/
│   ├── Datasets and Numpy Arrays/
│   │   ├── adasyn_mlp_x_train.csv
│   │   ├── adasyn_mlp_y_train.csv
│   │   ├── ae_mlp_x_test.csv
│   │   ├── ae_mlp_y_test.csv
│   │   ├── ae_X_train.csv
│   │   ├── ae_X_val.csv
│   │   ├── ae_Y_train.csv
│   │   ├── ae_y_train.csv
│   │   ├── ae_y_val.csv
│   │   ├── ae_per_feature_thresholds.npy
│   │   ├── FULL_INT8_per_feature_Threshold.npy
│   │   ├── mlp_x_val.csv
│   │   └── mlp_y_val.csv
│   │
│   ├── Models/
│   │   ├── Best_AE_float32.tflite
│   │   ├── Best_AE_fp16_weights.tflite
│   │   ├── Best_AE_int8_full.tflite
│   │   ├── Best_AE_int8_weights.tflite
│   │   ├── Best_AE.keras
│   │   ├── Best_MLP_float32.tflite
│   │   ├── Best_MLP_fp16_weights.tflite
│   │   ├── Best_MLP_int8_full.tflite
│   │   ├── Best_MLP_int8_weights.tflite
│   │   └── Best_MLP.keras
│   │
│   ├── FP32_AE_MLP_Model.ipynb
│   ├── Experiment_2_Quant_AE_MLP_Models.ipynb
│   ├── AE_MLP_Experiment_1_Resource_Usage.ipynb
│   ├── AE_MLP_Experiment_3_Generalisation_Bots.ipynb
│   └── AE_MLP_Experiment_3_Generalisation_DoS.ipynb



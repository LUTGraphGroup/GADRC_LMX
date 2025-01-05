# GADRC: Drug repositioning by graph convolution-attention encoder and deep residual networks combined with weighted cross-entropy

# Requirements:
- python 3.9.19          
- pandas 2.0.3
- cudatoolkit 11.3.1        
- pytorch  2.3.0
- numpy 1.23.5      
- scikit-learn  1.2.2
# Data:
The data files needed to run the model, which contain C-dataset and F-dataset.
- DrugFingerprint, DrugGIP: The similarity measurements of drugs to construct the similarity network
- DiseasePS, DiseaseGIP: The similarity measurements of diseases to construct the similarity network
- DrugDiseaseAssociationNumber: The known drug disease associations
# Code:
- data_preprocess.py: Methods of data processing
- metric.py: Metrics calculation
- train_DDA.py: Train the model

# Usage:
Execute ```python main.py``` 

### This is the instructions for running the model.

#### This project contains the following directories and files:
1. data/cropharvest-crop-detection: contains train and test datasets.
2. notebook: codes for processing the dataset and grid-searching on the random-forest model; visualizes the model's performance with a confusion matrix and ROC curve.
3. random_forest.py: code that uses the best hyper-parameters to train a random forest and save the test set results.
4. predictions: results will be saved into this directory.


Note 1: For both models, put train.csv and test_nolabels.csv files into a "data/cropharvest-crop-detection" directory; the results for the model will be saved in the "predictions" directory.

#### Installing packages
pip install -r requirements.txt

#### Running the Random Forest: 
python random_forest.py


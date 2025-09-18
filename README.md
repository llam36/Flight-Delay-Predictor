# File Directory Information:
All github pages files are located in the root(/) directory. You can find all files related to the datasets as well as the machine learning models in the Code directory.

## File Descriptions:
`/root/` - Files related to website generation

`/root/Code/` - Directory w/ all ML related code and data

`/root/Code/raw` - Raw data acquired from data source

`/root/Code/batched` - Data that has been batched to reduce load on memory

`/root/Code/create_batches.ipynb` - Creates 5 batches of data

`root/Code/combine_batches.ipynb` - Combines batches of data

`root/Code/preprocessing.ipynb` - Preprocessing code for data

`root/Code/random_forest.ipynb` - Random Forest Implementation

### Logistic Regression

`root/Code/Logistic/Logistic\ Regression\ Raw.ipynb` - Logistic Regression w/out Preprocessing Implementation

`root/Code/Logistic/Logistic\ Regression\ Original.ipynb` - Logistic Regression w/ Preprocessing Implementation

`root/Code/Logistic/Logistic\ Regression\ Without\ Top\ Features.ipynb` - Logistic Regression w/ Top Features Excluded Implementation

`root/Code/Logistic/models/logreg_model_raw.pkl` - Logistic Regression w/out Preprocessing

`root/Code/Logistic/models/logreg_model_original.pkl` - Logistic Regression w/ Preprocessing

`root/Code/Logistic/models/logreg_model_w_out_top_features.pkl` - Logistic Regression w/ Top Features Excluded

### Random Forest

`root/Code/random_forest.ipynb` - Random Forest with Preprocessing and Chi-Squared Feature Selection

`root/Code/random_forest_all.ipynb` - Random Forest with Preprocessing and No Feature Selection

`root/Code/random_forest_mutual.ipynb` - Random Forest with Preprocessing and Mutual Information Feature Selection

### Neural Network

`root/Code/NeuralNetwork.ipynb` - Neural Network script for model without feature selection, model with Chi-Squared Feature Selection, and model with Mutual Info feature selection.

`root/Code/NN_all_features.h5` - Neural Network trained on all features.

`root/Code/NN_chi2.h5` - Neural Network trained on chi-squared features.

`root/Code/NN_mutual_info.h5` - Neural Network trained on features with highest mutual info.

# To run docker container:

- docker-compose up --build
- Go to http://localhost:4000 to see your live site!

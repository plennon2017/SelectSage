# SelectSage
### Dataset
Make sure that the dataset csv files, and the features csv file all are saved in the NB15 directory in the provided milestone folder to run the latest codes.
### Creating Graphs
Run the script “data_pipeline.py” to create graphs for both all features and binary features. These graphs will be saved in the “graphs” directory. The train, test and validation split for both graphs is same.
### Hyperparameter Tuning for Similarity Thresholds and Functions
Check the configuration file “hyperparam_tuning_config.json” and change parameters of similarity threshold and similarity functions for different trials. Once the configuration file is ready then run the script “similarity_hyperparam_tuning.py” in order to train the classifier on the graph with every possible combination of similarity threshold and similarity function. The results of each possible trial will be saved in a csv file in the “hyperparam_tuning_results” directory. The plot can be made from this saved result by using the notebook “SelectSage_Results_2023.ipynb”.
### Performing Different Experiments
Check the configuration files for each type of experiment you want to perform in the “config” directory. Once the configuration files are ready then you can specify them inside the script “training.py” to perform the experiment related with that specific configuration file. Executing the script “training.py” will save the metrics, graph model and classifier model parameters as pickle objects in the subdirectory of “logs” directory. The plot can be made from this saved metrics pickle object by using the notebook “SelectSage_Results_2023.ipynb”.

import preprocessing
import naive_bayes
import bootstrapping

reduced_data, column_names, classifications = preprocessing.run("lifeexpecdata.csv")
#naive_bayes.training_validation_testing(reduced_data, classifications)
bootstrapping.run(reduced_data, classifications)
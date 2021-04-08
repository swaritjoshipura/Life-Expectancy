

Goal: To use ML algorithms to determine what factors contribute to longer life expectancy



We will use The World Health Organization’s on factors influencing Life Expectancy dataset for 193 countries from the years 2000-2015. The dataset has 22 features and 3,088 samples. We may have to shorten the number of features with PCA and remove some countries from the dataset. We plan to add a feature that classifies each sample of the dataset based on the life expectancy. Each sample will be assigned a certain classifier(e.g. 1, -1, 0, etc.) depending on the life expectancy range it falls in. 

Making use of: 
a) K-Nearest Neighbor
b) Naive-Bayes


We will apply training/testing/validation and bootstrapping by initially splitting the data up into different sets. For training/testing/validation we will split up the data into three datasets and each will hold approximately ⅓ of the samples, which will be assigned randomly.


For Bootstrapping, we will make the training dataset around 25% of the original dataset to allow for more generalization and use β=50 to get a good perspective of the overall error in case the errors produced have a lot of variance. We can apply these techniques for k-values ranging from 1-10 for the k-nearest neighbors algorithm and Naive Bayes doesn’t have any hyperparameters that we have to worry about.


We will show a plot of the amount of variance in the data described by the different number of features. We will show a plot of the errors for the different k-values (hyperparameters) of the k-nearest neighbor algorithm and the output of the error for the Naive-Bayes algorithm from training/validation/testing and bootstrapping. We will also show a ROC curve by varying the value of k for the k-nearest neighbor algorithm and calculating the area under the curve to get a different perspective of the usefulness of the k-nearest neighbor algorithm. We can’t implement a ROC curve for Naive-Bayes since it doesn’t have any hyperparameters.

## My Project <img width="250" height="250" align="right" src="/assets/IMG/new_mine.png">

I applied machine learning techniques to investigate mine detection and mine type classification.

## Introduction 

For both post- and active battle landmarks, minefields remain a problem for the soldiers and civilians present in the area. Landmines are used as a deterrent for both infantry and vehicles, and therefore mine detection remains an important feat for military efforts. However, in regions that were previously war-torn, minefields are still leftover once civilians begin to reoccupy the area. This often leads to the deaths of innocent people who wander out, and so the  majority of mine detection efforts are utilized to make these areas livable without danger again.

Several methods are currently used for detection. A more recent and unique application to the problem is the usage of rats to find landmines. Due to their lightweight, high mobility, and sensitive smell, they can be trained to identify likely spots for which landmines have been buried. Specialized vehicles that resist high explosive blasts can also be driven over fields to activate the mines, but this form of disarmament leads to disruption of the surrounding land and still poses some risks, especially when dealing with the stronger landmines. In terms of manual removal, there are two main employed features of detection: active and passive. In active detection, a sensor sends a signal into the ground, and receives a signal back from the mine. This form of detection often leads to the detonation of mines since the signal sent out triggers their explosives, which inherently poses a risk to the land and operators. In passive detection, a sensor simply receives signals from landmines. Although this avoids the issue of activation, this type of detection is not as effective. In mine discovery, extremely high accurracies are necessary as mistakes can lead to cost lives.

[This dataset](./assets/Mine_Dataset.csv) consists of voltage caused by distortion from the mine, height of the sensor to the ground, and the soil type around the buried mine (Figure 1). The mine types encountered at these respective conditions has been tabulated. This allows for a supervised, classification machine learning approach. Since the dataset is labelled and we know what correlates based on the input, the choice of method should be supervised. Since the output is one of five choices, the choice of methods should be a classification. Given the dataset's size of 338 instances, approaches such as decision trees, support vector classification, and ensemble methods are most appropriate.

<img width="800" height="500" align="center" src="/assets/IMG/table_descriptor.png">

*Figure 1: Parameters and labeling of mine type dataset. Retrieved from UCI Machine Learning Repository [1](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8443331)*

The employment of these methods attempts to provide a model that can accurately predict both the presence of, and type, of mine. Therefore, compounded by field knowledge or the usage of other assisted methods, passive detection can be improved which avoids landmine detonation.

## Data

The dataset was retrieved from the UCI Machine Learning Repository, and was originally sourced in the study "Passive Mine Detection and Classification Method Based on Hybrid Model" by Cemal Yilmaz, Hamdi Tolga Kahraman, and Salih Söyler[1](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8443331). Figure 2 shows the distribution of mine types within the data. The dataset is almost perfectly balanced between the mine types, with only a 9% difference (6 samples) between the most and least populated types. The data has also come pre-normalized, such that all values of voltage, height, and soil type exist as values between 0 and 1 inclusive.

<img width="700" height="500" align="center" src="/assets/IMG/mine_counts.png">

*Figure 2: Count of each instance for the respective mine types, as indicated in Fig. 1. The dataset is balanced.*

## Modelling

Given the nature of the problem set, machine learning models such as decision trees and support vectors are the most appropriate. In general, these can be improved with the K-Nearest Neighbors or various ensemble methods as well. In this report, these models were utilized as well as comparisons to other models that would normally be deemed non-classical for this dataset, such as a neural network.

The separation and processing of the data for modelling is as follows:

```python
X_data=mines.drop(['M'],axis=1).values #columns except for M
Y_data=mines['M'].values #M column
Y_data = Y_data.reshape(-1,1)

Y_binary_data = (Y_data >= 2).astype(int)

#split data into training and testing
test_size = 0.1
X_train, X_test, Y_train, Y_test, Y_binary_train, Y_binary_test = train_test_split(X_data, Y_data, Y_binary_data, test_size=test_size, random_state=3)
```
The X_data stores the V, H, and S categories, Y_data stores values from 1-5 depending on the mine type, and Y_binary_data converts Y_data into 0 or 1 depending on the absence of (0) or presense of (1) a landmine.

The general format utilized for each machine learning model for fitting the data is as follows:

```python
single_tree = DecisionTreeClassifier(max_depth=max_depth)
single_tree.fit(X_train, Y_train)

Y_pred_single = single_tree.predict(X_test).reshape(-1, 1)

print("Single Tree")
print(single_tree.feature_importances_)
single_acc = accuracy_score(Y_test, Y_pred_single)
single_score = single_tree.score(X_data, Y_data)
print(single_acc)
print(single_score)
ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred_single)
plt.show()
```

In the above implementation, a decision tree model is fit to the training data. The predicted values are then produced from the model. For the models that have the appropriate attributes, the feature importances were retrieved. Additionally, for each model, the accuracy with respect to the test data was calculated. In order to test the generalization of each model, the accuracy of the model against the entire dataset was also calculated. These accuracies were then represented via a confusion matrix.

## Results

Click [here](#summary) to jump to results summary and model comparison.

### Decision Tree

The decision tree utilized was set to have a maximum depth of 8.

<img width="500" height="500" align="center" src="/assets/IMG/decision_tree_binary_matrix.png">

*Figure 3: Confusion matrix of single decision tree (max depth of 8) for mine detection*

The decision tree predicts mine detection at an 88.2% accuracy for the testing data (Figure 3), which improves to 97.6% for the total dataset. For the features of voltage, height, and soil, the importances are 63.6%, 15.5%, and 20.8% respectively.

<img width="500" height="500" align="center" src="/assets/IMG/decision_tree_matrix.png">

*Figure 4: Confusion matrix of single decision tree (max depth of 8) for mine identification*

The decision tree predicts mine identification at a 55.9% accuracy for the testing data (Figure 4), which improves to 79.9% for the total dataset. For the features of voltage, height, and soil, the importances are 63.6%, 19.1%, and 17.3% respectively.

### Random Forest

The random forest decision trees were set to have a maximum depth of 8, with 100 estimators utilized.

<img width="500" height="500" align="center" src="/assets/IMG/random_forest_binary_matrix.png">

*Figure 5: Confusion matrix of random forest (each with max depth of 8) for mine detection*

The random forest predicts mine detection at an 88.2% accuracy for the testing data (Figure 5), which improves to 98.5% for the total dataset. For the features of voltage, height, and soil, the importances are 73.2%, 16.8%, and 1.0% respectively.

<img width="500" height="500" align="center" src="/assets/IMG/random_forest_matrix.png">

*Figure 6: Confusion matrix of random forest (each with max depth of 8) for mine identification*

The random forest predicts mine identification at a 61.7% accuracy for the testing data (Figure 6), which improves to 93.5% for the total dataset. For the features of voltage, height, and soil, the importances are 71.2%, 17.4%, and 11.4% respectively.

### Bagging

The bagging model utilized uses a decision tree classifier as its base estimator. Random subsets of data are taken in a bootstrap fashion (taken with replacement), and aggregated together into a final prediction.

<img width="500" height="500" align="center" src="/assets/IMG/bagging_binary_matrix.png">

*Figure 7: Confusion matrix of bagging for mine detection*

The bagging classifier predicts mine detection at a 91.2% accuracy for the testing data (Figure 7). For the total dataset, this improves to a 97.9% accuracy. 

<img width="500" height="500" align="center" src="/assets/IMG/bagging_matrix.png">

*Figure 8: Confusion matrix of bagging for mine identification*

The bagging classifier predicts mine identification at a 61.7% accuracy for the testing data (Figure 8). For the total dataset, this improves to a 92.9% accuracy. 

### AdaBoost

The adaboost classifier utilizes the aforementioned decision tree classifer as its base estimator. Adaboost improves upon more difficult cases by focusing on incorrectly classified instances over each new iteration.

<img width="500" height="500" align="center" src="/assets/IMG/adaboost_binary_matrix.png">

*Figure 9: Confusion matrix of adaboost for mine detection*

The adaboost classifier predicts mine detection at an 88.2% accuracy for the testing data (Figure 9). This improves to a 98.8% accuracy over the entire dataset.

<img width="500" height="500" align="center" src="/assets/IMG/adaboost_matrix.png">

*Figure 10: Confusion matrix of adaboost for mine identification*

The adaboost classifier predicts mine identification at a 55.9% accuracy for the testing data (Figure 10). This improves to a 95.6% accuracy over the entire dataset.

### Support Vector

The support vector classifier uses a one-to-one separation approach in this multiclassification case. The kernel utilized was the radial basis function (rbf).

<img width="500" height="500" align="center" src="/assets/IMG/svc_binary_matrix.png">

*Figure 11: Confusion matrix of support vector for mine detection*

The support vector classifier predicts mine detection at an 82.4% accuracy for the testing data (Figure 11). This decreases to a 79.0% accuracy over the entire dataset.

<img width="500" height="500" align="center" src="/assets/IMG/svc_matrix.png">

*Figure 12: Confusion matrix of support vector for mine identification*

The support vector classifier predicts mine identifcation at a 47.0% accuracy for the testing data (Figure 12). This improves to a 53.6% accuracy over the entire dataset.

### Neural Network

The neural network utilized consists of 2 hidden layers, each with 5 nodes.

<img width="500" height="500" align="center" src="/assets/IMG/neural_binary_matrix.png">

*Figure 13: Confusion matrix of neural network (2 hidden layers of 5 nodes each) for mine detection*

The neural network predicts mine detection at an 88.2% accuracy for the testing data (Figure 13). This decreases to a 86.4% accuracy over the entire dataset.

<img width="500" height="500" align="center" src="/assets/IMG/neural_matrix.png">

*Figure 14: Confusion matrix of neural network (2 hidden layers of 5 nodes each) for mine identification*

The neural network predicts mine identifcation at a 58.8% accuracy for the testing data (Figure 14). This decreases to a 51.8% accuracy over the entire dataset.

### K-Nearest Neighbors

The algorithm utilized weighted distance to determine neighbor importance.

<img width="500" height="500" align="center" src="/assets/IMG/knn_binary_matrix.png">

*Figure 15: Confusion matrix of k-nearest neighbors (weighted by distance) for mine detection*

The k-nearest neighbors predicts mine detection at a 70.6% accuracy for the testing data (Figure 15). This improves to a 97.0% accuracy over the entire dataset.

<img width="500" height="500" align="center" src="/assets/IMG/knn_matrix.png">

*Figure 16: Confusion matrix of k-nearest neighbots (weighted by distance) for mine identification*

The k-nearest neighbors predicts mine identification at a 44.1% accuracy for the testing data (Figure 16). This improves to a 94.4% accuracy over the entire dataset.

### Hard Voting

Voting classifier that takes in the random forest, bagging, adaboost, and k-nearest neighbors classifiers. Hard voting utilized such that majority decision wins.

<img width="500" height="500" align="center" src="/assets/IMG/voting_binary_matrix.png">

*Figure 17: Confusion matrix of hard voting (forest, bagging, ada, knn) for mine detection*

The voting classifier predicts mine detection at a 91.2% accuracy for the testing data (Figure 17).

<img width="500" height="500" align="center" src="/assets/IMG/voting_binary_all.png">

*Figure 18: Confusion matrix of hard voting (forest, bagging, ada, knn) for mine detection, extrapolated to all data*

The voting classifier predicts mine detection at a 99.1% accuracy for all data (Figure 18).

<img width="500" height="500" align="center" src="/assets/IMG/voting_matrix.png">

*Figure 19: Confusion matrix of hard voting (forest, bagging, ada, knn) for mine identification*

The voting classifier predicts mine identification with a 61.8% accuracy for testing data (Figure 19).

<img width="500" height="500" align="center" src="/assets/IMG/voting_matrix_all.png">

*Figure 20: Confusion matrix of hard voting (forest, bagging, ada, knn)for mine identification, extrapolated to all data*

The voting classifier predicts mine identification with a 95.9% accuracy for all data (Figure 19).

### Model Comparison <a id="summary"></a>

<img width="800" height="500" align="center" src="/assets/IMG/accuracy_test_binary.png">

*Figure 21: Compared accuracy of models for mine detection against test data*

<img width="800" height="500" align="center" src="/assets/IMG/accuracy_all_binary.png">

*Figure 22: Compared accuracy of models for mine detection against all data*

<img width="800" height="500" align="center" src="/assets/IMG/accuracy_test.png">

*Figure 23: Compared accuracy of models for mine identification against test data*

<img width="800" height="500" align="center" src="/assets/IMG/accuracy_all.png">

*Figure 24: Compared accuracy of models for mine identification against all data*

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8443331) Yilmaz, C., Kahraman, H. T., & Söyler, S. (2018). Passive mine detection and classification method based on hybrid model. IEEE Access, 6, 47870-47888.

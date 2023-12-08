## My Project <img width="250" height="250" align="right" src="/assets/IMG/new_mine.png">

I applied machine learning techniques to investigate mine detection and mine type classification.

## Introduction 

For both post- and active battle landmarks, minefields remain a problem for the soldiers and civilians present in the area. Landmines are used as a deterrent for both infantry and vehicles, and therefore mine detection remains an important feat for military efforts. However, in regions that were previously war-torn, minefields are still leftover once civilians begin to reoccupy the area. This often leads to the deaths of innocent people who wander out, and so the majority of mine detection efforts are utilized to make these areas livable without danger again.

Several methods are currently used for detection. A more recent and unique application to the problem is the usage of rats to find landmines. Due to their lightweight, high mobility, and sensitive smell, they can be trained to identify likely spots for which landmines have been buried. Specialized vehicles that resist high explosive blasts can also be driven over fields to activate the mines, but this form of disarmament leads to disruption of the surrounding land and still poses some risks, especially when dealing with the stronger landmines. In terms of manual removal, there are two main employed features of detection: active and passive. In active detection, a sensor sends a signal into the ground, and receives a signal back from the mine. This form of detection often leads to the detonation of mines since the signal sent out triggers their explosives, which inherently poses a risk to the land and operators. In passive detection, a sensor simply receives signals from landmines. Although this avoids the issue of activation, this type of detection is not as effective. In mine discovery, extremely high accuracies are necessary as mistakes can lead to cost lives. 

<img width="350" height="200" align="right" src="/assets/IMG/Mine-detection-methods.png">

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
The X_data stores the V, H, and S categories, Y_data stores values from 1-5 depending on the mine type, and Y_binary_data converts Y_data into 0 or 1 depending on the absence of (0) or presence of (1) a landmine.

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

The adaboost classifier utilizes the aforementioned decision tree classifier as its base estimator. Adaboost improves upon more difficult cases by focusing on incorrectly classified instances over each new iteration.

<img width="500" height="500" align="center" src="/assets/IMG/adaboost_binary_matrix.png">

*Figure 9: Confusion matrix of adaboost for mine detection*

The adaboost classifier predicts mine detection at an 88.2% accuracy for the testing data (Figure 9). This improves to a 98.8% accuracy over the entire dataset.

<img width="500" height="500" align="center" src="/assets/IMG/adaboost_matrix.png">

*Figure 10: Confusion matrix of adaboost for mine identification*

The adaboost classifier predicts mine identification at a 55.9% accuracy for the testing data (Figure 10). This improves to a 95.6% accuracy over the entire dataset.

### Support Vector

The support vector classifier uses a one-to-one separation approach in this multi-classification case. The kernel utilized was the radial basis function (rbf).

<img width="500" height="500" align="center" src="/assets/IMG/svc_binary_matrix.png">

*Figure 11: Confusion matrix of support vector for mine detection*

The support vector classifier predicts mine detection at an 82.4% accuracy for the testing data (Figure 11). This decreases to a 79.0% accuracy over the entire dataset.

<img width="500" height="500" align="center" src="/assets/IMG/svc_matrix.png">

*Figure 12: Confusion matrix of support vector for mine identification*

The support vector classifier predicts mine identification at a 47.0% accuracy for the testing data (Figure 12). This improves to a 53.6% accuracy over the entire dataset.

### Neural Network

The neural network utilized consists of 2 hidden layers, each with 5 nodes.

<img width="500" height="500" align="center" src="/assets/IMG/neural_binary_matrix.png">

*Figure 13: Confusion matrix of neural network (2 hidden layers of 5 nodes each) for mine detection*

The neural network predicts mine detection at an 88.2% accuracy for the testing data (Figure 13). This decreases to a 86.4% accuracy over the entire dataset.

<img width="500" height="500" align="center" src="/assets/IMG/neural_matrix.png">

*Figure 14: Confusion matrix of neural network (2 hidden layers of 5 nodes each) for mine identification*

The neural network predicts mine identification at a 58.8% accuracy for the testing data (Figure 14). This decreases to a 51.8% accuracy over the entire dataset.

### K-Nearest Neighbors

The algorithm utilized weighted distance to determine neighbor importance.

<img width="500" height="500" align="center" src="/assets/IMG/knn_binary_matrix.png">

*Figure 15: Confusion matrix of k-nearest neighbors (weighted by distance) for mine detection*

The k-nearest neighbors predicts mine detection at a 70.6% accuracy for the testing data (Figure 15). This improves to a 97.0% accuracy over the entire dataset.

<img width="500" height="500" align="center" src="/assets/IMG/knn_matrix.png">

*Figure 16: Confusion matrix of k-nearest neighbors (weighted by distance) for mine identification*

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

Bar graphs were created for comparison between all models and their accuracies to the respective datasets.

<img width="800" height="500" align="center" src="/assets/IMG/accuracy_test_binary.png">

*Figure 21: Compared accuracy of models for mine detection against test data*

<img width="800" height="500" align="center" src="/assets/IMG/accuracy_all_binary.png">

*Figure 22: Compared accuracy of models for mine detection against all data*

Against the test data set, the best models for mine detection were hard voting and bagging, and the least effective model was k-nearest neighbors. However, against the entire dataset, voting had the highest success rate whilst the support vector and neural network failed to meet the 90% mark.

<img width="800" height="500" align="center" src="/assets/IMG/accuracy_test.png">

*Figure 23: Compared accuracy of models for mine identification against test data*

<img width="800" height="500" align="center" src="/assets/IMG/accuracy_all.png">

*Figure 24: Compared accuracy of models for mine identification against all data*

Against the test data set, the best models for mine identification were hard voting, bagging, and random forest. However, against the entire dataset, voting and adaboost had the highest success, whereas the support vector and neural network barely predicted over 50%.

## Discussion

Looking first into simply mine detection, all models did fairly well both in the test case and against the whole dataset (Figures 21 & 22). The vast improvement in detection over identification stems from the fact that the problem is reduced to a binary solution. One issue that arises however is that mine class 1 corresponds to no mine, whereas mine class 2-5 corresponds to a mine. Therefore, the binary dataset becomes imbalanced, which skews and biases the predictions, which could be the primary reason for the neural network's decreased performance. The support vector also did not extrapolate well to the whole dataset, which is most likely due to a similar reason in which the boundary characterized for the test case ended up not working for the whole dataset in which there were an even greater increase in mine-labeled points. The decision tree based models (random forest, adaboost, bagging, and voting) all performed extremely well, with a 99.1% success rate for voting (Figure 22). Looking at the parameter importances, it seems that for just detection, the voltage received from the passive detector when the magnetic field is modified gives the biggest indicator at around the 60-70% range. This follows as the presence of a mine should be causes these field changes, whereas no mine would cause no field change. The other parameters of height and soil type should not matter as much for just the detection of a mine. As mentioned before, the indication of voting have the highest success rate supports the hypothesis that the best way to approach mine detection and identification is to utilize several models combined with field knowledge. In this way, the errors of each can be picked up and improved upon by the next steps, leading to almost perfect predictions.

As for the problem of mine-specific identification, all of the models performed relatively poor, with some falling even below random guessing (Figure 23). Looking at the confusion matrices that correspond to these test predictions, it appears that most of the models can accurately predict mine classes 1 and 2, however they have trouble when it comes to 3, 4 and 5. For instance, adaboost predicts case 1 and case 2 with a 66.7% and 90.0% success rate respectively. However for cases 3, 4, and 5, this falls to 33.3%, 25.0%, and 42.9% respectively (Figure 10). This may be partially due to the sparsity of data in those categories for the specific random seed utilized to test, however as established prior, it is already easy to determine no mine vs. mine, which explains the success in predicting case 1. Case 2 may have the parameters more closely correlated to its output and could be the easier mine out of the bunch to pick out. Additionally, the models may have not been fully optimized and could be overfitting to the training data, which causes difficulty in predicting the test cases when they deviate. Cases 1 and 2 may not deviate as much from training data, and so further continuation of this project should focus on optimization of the models and ensuring proper fitting.

When scored against the entire dataset, many of the models do much better except for the neural network and support vector. The greatest increase in success came from the k-nearest neighbors approach which improved to 94.4% from 44.1%. Given that this approached is based on neighbor data point proximity and was weighted by that distance, this increase is most likely due to the model now having more points nearby one another and therefore being able to make better predictions based on the clustering. Similar to the detection, identification also has the greatest success with the hard voting approach at 95.9% (Figure 24). This also supports the hypothesis in the same way that multiple models compounded by preexisting knowledge will give the best success rate at identifying mine types. Bagging and adaboost also saw marked increases which is most likely explained from the algorithms sampling process. Bagging can draw random subsets from the main set, which helps prevent its limitation to categories that may be lacking points as they can be sampled multiple times. Similarly adaboost runs through multiple iterations to reflect upon incorrect predictions, and reparameterizes to attempt to better focus on hitting those error/edge cases as it goes through. Surprisingly, the neural network and support vector still did not perform well even with the entire dataset. Both had approximately percentages equivalent to random guessing (Figure 24). For the neural network, it is very likely that it overfitted to the training data, and therefore did not extrapolate well to the dataset. Since it is a blackbox approach, it is hard to tell where the network may be faltering, however it could also be an optimization problem with the amount of hidden layers and nodes in each of them. As for the support vector, since the identification problem is multiclassed, the one-to-one approach that the support vector has to take to make a boundary in the hyperplane probably does not extend well to the larger dataset when there are many more points established. If the dataset was more clearly separated, this approach would have performed better, however it seems many of the mine types have similar characteristics which makes it hard to delineate them by linear/planar correlation. 

## Conclusion

Mine detection and identification remains a pressing issue in presently and previously war-torn areas, for both the military and civilians. In order to avoid accidental detonation, passive methods have to be utilized that often lack the same success rate as active methods. As demonstrated in this report, by employing machine learning models to the data collected by passive detectors, the presence or absence of a mine can be detected with very high accuracy, and the specific identification of a mine can be predicted with high accuracy as well. The most effective method is that which combines several methods into one, performing at a 99.1% accuracy for detection and 95.9% accuracy for identification. If only one method could be selected, AdaBoost provides the best opportunity for success in each category at a 98.8% accuracy for detection and 95.6% accuracy for identification. Given that the most separating parameter is only that of voltage, adaboost effectively identifies the more convoluted data points through several iterations of reoptimization that differ by the weaker parameters such as height or soil type.

In the future, more data points could be collected to expand the dataset and thereby further improve the models' accuracies. Since the parameter of greatest importance is that of voltage caused by magnetic distortion, it should be looked into to develop detectors that focus on fine measurement of these voltage differences such that the models can be redeveloped to pinpoint differences between datapoints. This report establishes that with the assistance of machine learning models, passive detectors can accurately detect and identify landmines, bypassing the need and risk for active methods that risk lies. This could lead to better removal plans of known mine fields and help prevent tragic loss of military, civilians, and assisting animals.

## References
[1](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8443331) Yilmaz, C., Kahraman, H. T., & Söyler, S. (2018). Passive mine detection and classification method based on hybrid model. IEEE Access, 6, 47870-47888.

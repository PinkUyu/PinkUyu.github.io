## My Project <img width="250" height="250" align="right" src="/assets/IMG/new_mine.png">

I applied machine learning techniques to investigate... Below is my report.

## Introduction 

Here is a summary description of the topic. Here is the problem. This is why the problem is important.

There is some dataset that we can use to help solve this problem. This allows a machine learning approach. This is how I will solve the problem using supervised/unsupervised/reinforcement/etc. machine learning.

<img width="800" height="500" align="center" src="/assets/IMG/table_descriptor.png">

*Figure 1: Parameters and labeling of mine type dataset. Retrieved from UCI Machine Learning Repository[1]*

We did this to solve the problem. We concluded that...

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

<img width="700" height="500" align="center" src="/assets/IMG/mine_counts.png">

*Figure 2: Count of each instance for the respective mine types, as indicated in Fig. 1. The dataset is balanced.*

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] Yilmaz, C., Kahraman, H. T., & Söyler, S. (2018). Passive mine detection and classification method based on hybrid model. IEEE Access, 6, 47870-47888.

[back](./) 

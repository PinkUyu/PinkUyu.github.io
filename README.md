## My Project <img width="250" height="250" align="right" src="/assets/IMG/new_mine.png">

I applied machine learning techniques to investigate mine detection and mine type classifcation.

## Introduction 

For both post- and active battle landmarks, minefields remain a problem for the soldiers and civilians present in the area. Landmines are used as a deterrant for both infrantry and vehicles, and therefore mine detection remains an important feat for military efforts. However, in regions that were previously war-torn, minefields are still leftover once civilians begin to reoccupy the area. This often leads to the deaths of innocent people who wander out, and so the  majority of mine detection efforts are utilized to make these areas livable without danger again.

Several methods are currently used for detection. A more recent and unique application to the problem is the usage of rats to find landmines. Due to their lightweight, high mobility, and sensitive smell, they can be trained to identify likely spots for which landmines have been buried. Specialized vehicles that resist high explosive blasts can also be driven over fields to activate the mines, but this form of disarmament leads to disruption of the surrounding land and still poses some risks, especially when dealing with the stronger landmines. In terms of manual removal, there are two main employed features of detection: active and passive. In active detection, a sensor sends a signal into the ground, and receives a signal back from the mine. This form of detection often leads to the detonation of mines since the signal sent out triggers their explosives, which inherently poses a risk to the land and operators. In passive detection, a sensor simply receives signals from landmines. Althought this avoids the issue of activation, this type of detection is not as effective. In mine discovery, extremely high accuracies are necessary as mistakes can lead to cost lives.

[This dataset](./assets/Mine_Dataset.csv) consists of voltage caused by distortion from the mine, height of the sensor to the ground, and the soil type around the buried mine (Figure 1). The mine types encountered at these respective conditions has been tabulated. This allows for a supervised, classifcation machine learning approach. Since the dataset is labelled and we know what correlates based on the input, the choice of method should be supervised. Since the output is one of five choices, the choice of methods should be a classifcation. Given the dataset's size of 338 instances, approaches such as decision trees, support vector classifcation, and ensemble methods are most appropriate.

<img width="800" height="500" align="center" src="/assets/IMG/table_descriptor.png">

*Figure 1: Parameters and labeling of mine type dataset. Retrieved from UCI Machine Learning Repository [1](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8443331)*

The employment of these methods attempts to provide a model that can accurately predict both the presence of, and type, of mine. Therefore, compounded by field knowledge or the usage of other assisted methods, passive detection can be improved which avoids landmine detonation.

## Data

The dataset was retrieved from the UCI Machine Learning Repository, and was originally sourced in the study "Passive Mine Detection and Classification Method Based on Hybrid Model" by Cemal Yilmaz, Hamdi Tolga Kahraman, and Salih Söyler.[1](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8443331)

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
[1](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8443331) Yilmaz, C., Kahraman, H. T., & Söyler, S. (2018). Passive mine detection and classification method based on hybrid model. IEEE Access, 6, 47870-47888.

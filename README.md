**Hi class, welcome to the AOS C111/204 final project!** <img align="right" width="220" height="220" src="/assets/IMG/new_mine.png">

For this project, you will be applying your skills to train a machine learning model using real-world data, then publishing a report on your own website.

* To get data for your project, you could:
  * use **your own data** from a separate research activity
  * **scour the internet** to find something original, then preprocess it yourself - see the Module Overview on BruinLearn for some resources
  * browse an archive of data designed for machine learning problems, such as the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/datasets)

* Your report should be in the region of 2000-2500 words with three to four figures, and written in a scientific language and style. [This template page](/project.md) gives an example structure that you could use, but feel free to make it your own.

Your website will be a great addition to your CV, and a place to host future projects too since it doubles as a GitHub repository. The first step is to set up a project website like this one by following the instructions below. 

## My Project

I applied machine learning techniques to investigate... Below is my report.

***

## Introduction 

Here is a summary description of the topic. Here is the problem. This is why the problem is important.

There is some dataset that we can use to help solve this problem. This allows a machine learning approach. This is how I will solve the problem using supervised/unsupervised/reinforcement/etc. machine learning.

We did this to solve the problem. We concluded that...

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

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
[1] DALL-E 3

[back](./) 

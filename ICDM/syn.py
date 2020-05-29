from ds_classifier import DSClassifier
from skmultiflow.trees.hoeffding_tree import HoeffdingTree

classifier = DSClassifier(learner=HoeffdingTree)
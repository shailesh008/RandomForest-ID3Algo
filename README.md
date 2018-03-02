# RandomForest-ID3Algo
Machine Learning.Building Random Forest &amp; decision tree Using ID3 Algorithm using Breast Cancer &amp; Car dataSet.

  Training Phase/Testing Phase:

Training Phase: 
Begin with the original set of attributes as the root node.
On each iteration of the algorithm, we iterate through every unused attribute of the remaining set and calculates the entropy (or information gain) of that attribute.
Then, select the attribute which has the smallest entropy (or largest information gain) value.

TestingPhase:
Uses a greedy approach by selecting the best attribute to split the dataset on each iteration
Accuracy is better when discrete splitting and random shuffling of data is used along with this algorithm.

For Breast Cancer dataSet:------------------------------------------------------------------------------------------

Building ID3 Decision Tree with Number of records: 684
('Training-set data length :', 615)
('Test-set data length :', 69)
***********************************----ID3---*******************************************
Accuracy of training set is: 100.0000
Accuracy of test set is: 92.7536
********************************------Random Forest-------********************************
RandomForest Accuracy of training set is: 95.9350
RandomForest Accuracy of test set is: 94.2029

For Car DataSet:------------------------------------------------------------------------------------------------------

Building ID3 Decision Tree with Number of records: 1729
('Training-set data length :', 1383)
('Test-set data length :', 346)
**************************----ID3---*******************************************
Accuracy of training set is: 100.0000
Accuracy of test set is: 57.2254
***********************----Random Forest--*************************************
RandomForest Accuracy of training set is: 88.8648
RandomForest Accuracy of test set is: 60.1156




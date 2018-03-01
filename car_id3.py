import csv
import math
import random
from collections import Counter

# Implement your decision tree below
# Used the ID3 algorithm to implement the Decision Tree

# ID3DecTree Class used for learning and building the Decision Tree using the given Training Set
# Built decision tree is tested on the test data with Data select ,shuffle   & replace .
class ID3DecTree():
    tree = {}

    def learn(self, training_set, attributes, target):
        self.tree = build_tree(training_set, attributes, target)


# Class Node which will be used while classify a test-instance using the tree which was built earlier
class Node():
    value = ""
    children = []

    def __init__(self, val, dictionary):
        self.value = val
        if (isinstance(dictionary, dict)):
            self.children = dictionary.keys()


# Majority Function which tells which class has more entries in given data-set
def majorClass(attributes, data, target):

    freq = {}
    index = attributes.index(target)

    for tuple in data:
        if (freq.has_key(tuple[index])):
            freq[tuple[index]] += 1
        else:
            freq[tuple[index]] = 1

    max = 0
    major = ""

    for key in freq.keys():
        if freq[key]>max:
            max = freq[key]
            major = key

    return major


# Calculates the entropy of the data given the target attribute
def entropy(attributes, data, targetAttr):

    freq = {}
    dataEntropy = 0.0

    i = 0
    for entry in attributes:
        if (targetAttr == entry):
            break
        i = i + 1

    i = i - 1

    for entry in data:
        if (freq.has_key(entry[i])):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]]  = 1.0

    for freq in freq.values():
        dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2)

    return dataEntropy


# Calculates the information gain (reduction in entropy) in the data when a particular
# attribute is chosen for splitting the data.
def info_gain(attributes, data, attr, targetAttr):

    freq = {}
    subsetEntropy = 0.0
    i = attributes.index(attr)

    for entry in data:
        if (freq.has_key(entry[i])):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]]  = 1.0

    for val in freq.keys():
        valProb        = freq[val] / sum(freq.values())
        dataSubset     = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)

    return (entropy(attributes, data, targetAttr) - subsetEntropy)


# This function chooses the attribute among the remaining attributes which has the maximum information gain.
def attr_choose(data, attributes, target):

    best = attributes[0]
    maxGain = 0;

    for attr in attributes:
        newGain = info_gain(attributes, data, attr, target)
        if newGain>maxGain:
            maxGain = newGain
            best = attr

    return best


# This function will get unique values for that particular attribute from the given data
def get_values(data, attributes, attr):

    index = attributes.index(attr)
    values = []

    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])

    return values

# This function will get all the rows of the data where the chosen "best" attribute has a value "val"
def get_data(data, attributes, best, val):

    new_data = [[]]
    index = attributes.index(best)

    for entry in data:
        if (entry[index] == val):
            newEntry = []
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            new_data.append(newEntry)

    new_data.remove([])
    return new_data
# This function used to calculte the accuracy of the tree over training & test Set
def accuracy_calculation(tree,training_set,test_set,target,attributes):
    results = []
    for entry in test_set:
        tempDict = tree.tree.copy()
        result = ""
        while(isinstance(tempDict, dict)):
            root = Node(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])
            tempDict = tempDict[list(tempDict.keys())[0]]
            index = attributes.index(root.value)
            value = entry[index]
            if(value in tempDict.keys()):
                child = Node(value, tempDict[value])
                result = tempDict[value]
                tempDict = tempDict[value]
            else:
                result = "Null"
                break
        results.append(result == entry[-1])
    accuracy = float(results.count(True))/float(len(results))*100
    return accuracy

#id3_algorithm will generate tree and test it on test_dataset and calculate accuracy based on it

def id3_algorithm(training_set,test_set,target,attributes):

    tree = ID3DecTree()
    tree.learn(training_set, attributes, target )

    return accuracy_calculation(tree,training_set,test_set,target,attributes)

# This function is used to build the decision tree using the given data,
# attributes and the target attributes. It returns the decision tree in the end.
def build_tree(data, attributes, target):

    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    default = majorClass(attributes, data, target)

    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        best = attr_choose(data, attributes, target)
        tree = {best:{}}

        for val in get_values(data, attributes, best):
            new_data = get_data(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = build_tree(new_data, newAttr, target)
            tree[best][val] = subtree

    return tree




#This function will build random forest with 10 trees in it
#Accuracy of each tree and average of all is the output of this tree
def randomForest(training_set, test_set,target,attributes):
    K = 3
    trees =[]
    for k in range(K):
        random.shuffle(training_set)
        train=[]
        for i in range(0,int(len(training_set)/2)):
            train.append(training_set[i])
        tree = ID3DecTree()
        tree.learn(train, attributes, target)
        trees.append(tree)
    acc = []
    results = []
    test =[]
    '''for i in range(0,100):
        test.append(test_set[i])'''
    for entry in test_set:
        att_acc=[]
        #total_result =0
        for tree in trees:
            tempDict = tree.tree.copy()
            result = ""
            while(isinstance(tempDict, dict)):
                root = Node(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])
                tempDict = tempDict[list(tempDict.keys())[0]]
                index = attributes.index(root.value)
                value = entry[index]
                if(value in tempDict.keys()):
                    child = Node(value, tempDict[value])
                    result = tempDict[value]
                    tempDict = tempDict[value]
                else:
                    result = "Null"
                    break

            att_acc.append(result == entry[-1])
        #print(att_acc)
        count = Counter(att_acc)
        count_most = count.most_common(1)[0][0]
        results.append(count_most)
        #print(results)

    avg_acc = float(results.count(True))/len(test_set)*100
    #print ("Average accuracy of random forest is : %.4f" % avg_acc)
    return avg_acc


# This function runs the decision tree algorithm. It parses the file for the data-set, and then it runs the 10-fold cross-validation. It also classifies a test-instance and later compute the average accurracy
# Improvements Used:

# Generate Multiple Trees with random quantity of dataset
def build_random_Forest_decision_tree():

    data = []

    with open("car.csv") as csvfile:
        for line in csv.reader(csvfile, delimiter=','):



            data.append(line)

	print "Building ID3 Decision Tree with Number of records: %d" % len(data)

	#attributes = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-info_gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
        attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot','safety', 'result']


    #Divide the data into Training Set(80%) and Test Set(20%)
        #random.shuffle(data)
        training_set=[]
        for i in range(0,int(len(data)*0.8)):
            training_set.append(data[i])
        test_set = []
        for i in range(int(len(data)*0.8),len(data)):
            test_set.append(data[i])
        print('Training-set data length :',len(training_set))
        print('Test-set data length :',len(test_set))

	target = attributes[-1]

    #ID3 algorithm which calculate accuracy
        print('***********************************----ID3---*******************************************')
        accuracy1 = id3_algorithm(training_set,training_set,target,attributes)
        print('Accuracy of training set is: %.4f'%accuracy1)
        accuracy = id3_algorithm(training_set,test_set,target,attributes)
        print('Accuracy of test set is: %.4f'%accuracy)

    #Random forest function which generate 5 trees
        print('***********************************----Random Forest---*******************************************')
        randomForestAccuracy = randomForest(training_set,training_set,target,attributes)
        print('RandomForest Accuracy of training set is: %.4f'%randomForestAccuracy)
        randomForestAccuracy1 = randomForest(training_set,test_set,target,attributes)
        print('RandomForest Accuracy of test set is: %.4f'%randomForestAccuracy1)



    f = open("car_result.txt", "a+")
    #f.write("Random Forest Tree Accuracy: %.4f\n" % accuracy)
    f.write('Training-set data length %d:\n' % len(training_set))
    f.write('Test-set data length %d:\n' % len(test_set))
    f.write('Accuracy of test set is: %.4f\n'%accuracy)
    f.write('Accuracy of training set is: %.4f\n'%accuracy1)
    f.write('RandomForest Accuracy of training set is: %.4f\n'%randomForestAccuracy)
    f.write('RandomForest Accuracy of test set is: %.4f\n'%randomForestAccuracy1)
    f.close()



if __name__ == "__main__":
	build_random_Forest_decision_tree()

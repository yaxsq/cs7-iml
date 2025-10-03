# Question 1

# Step 0
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

print("\n\n  Question 1")

# Step 1  Loading
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv("iris.data", names=names)

# Step 2  Shape
print(dataset.shape)
# print(dataset.head)

# Step 3  Group
print(dataset.groupby('class').size())

# Step 4  train / test dataset split
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
t_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=t_size, random_state=seed)

# Step 5  kNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)

print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
print("\n")

# Step 6  changing neighbors
for i in range (1, 11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_test)

    print(f"K: {i}, {accuracy_score(Y_test, predictions)}")
print("\n")

# Step 7  changing seed
t_size = 0.20
for newSeed in range(1, 11):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=t_size, random_state=newSeed)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_test)
    print(f"Seed: {newSeed}, Accuracy: {accuracy_score(Y_test, predictions)}")


# PART 2
print("\n\n\n\n   Question 2")

from sklearn.preprocessing import StandardScaler

# Step 1  Loading
train_data = pd.read_csv("Occupency_Detection/datatraining.txt")
test_data = pd.read_csv("Occupency_Detection/datatest.txt")

# Step 2  Shape
print(train_data.shape)
print(test_data.shape)

# Step 3  Group
print(train_data.groupby('Occupancy').size())
print(test_data.groupby('Occupancy').size())

print(train_data['Occupancy'].value_counts())
print(test_data['Occupancy'].value_counts())

print(train_data.head())

# Step 4  Dropping index and date columns
X_train = train_data.drop(['date', 'Occupancy'], axis=1)
Y_train = train_data['Occupancy']
X_test = test_data.drop(['date', 'Occupancy'], axis=1)
Y_test = test_data['Occupancy']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5  kNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, Y_train)
predictions = knn.predict(X_test_scaled)

print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

# Step 6  changing neighbors
for i in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, Y_train)
    predictions = knn.predict(X_test_scaled)
    print(f"K: {i}, {accuracy_score(Y_test, predictions)}")


# PART 3
print("\n\n\n\n   Question 3")

data = [
    [1, 4, 1, 0],
    [2, 3, 5, 0],
    [1, 1, 1, 0],
    [1, 3, 4, 1],
    [1, 2, 3, 1]
]

# Input for the 6th row
print("Enter values for the 6th data point:")
a1 = int(input("Attribute 1: "))
a2 = int(input("Attribute 2: "))
a3 = int(input("Attribute 3: "))
data.append([a1, a2, a3, 1])

# Temporary placeholder values for last row
# data.append([2, 3, 4, 1])

X = [row[0:3] for row in data]
y = [row[3] for row in data]

print("Dataset:")
for i, row in enumerate(data):
    print(f"Point {i + 1}: {row[0:3]} -> Label {row[3]}")

# Manually creating stratified folds
folds = [
    [0, 3],
    [1, 4],
    [2, 5]
]


def manhattanDistance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += abs(point1[i] - point2[i])
    return distance


def getKNeighbors(X_train, y_train, testPoint, k):
    distances = []
    for i in range(len(X_train)):
        dist = manhattanDistance(X_train[i], testPoint)
        distances.append((dist, y_train[i]))

    # Sort by distance
    sorted_distances = []
    for dist, label in distances:
        inserted = False
        for j in range(len(sorted_distances)):
            if dist < sorted_distances[j][0]:
                sorted_distances.insert(j, (dist, label))
                inserted = True
                break
        if not inserted:
            sorted_distances.append((dist, label))

    return [label for (dist, label) in sorted_distances[:k]]


def predict(X_train, y_train, testPoint, k):
    neighbors = getKNeighbors(X_train, y_train, testPoint, k)

    # votes for distances
    vote_count = {}
    for label in neighbors:
        if label in vote_count:
            vote_count[label] += 1
        else:
            vote_count[label] = 1

    # Finding max vote
    max_votes = 0
    prediction = None
    for label, count in vote_count.items():
        if count > max_votes:
            max_votes = count
            prediction = label

    return prediction


def calculateMetrics(actual, predicted):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for a, p in zip(actual, predicted):
        if a == 1 and p == 1:
            tp += 1
        elif a == 0 and p == 0:
            tn += 1
        elif a == 0 and p == 1:
            fp += 1
        elif a == 1 and p == 0:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return tp, tn, fp, fn, accuracy, precision, recall, f1


allPredictions1 = []
allPredictions3 = []
allActuals = []

for fold_num, testFold in enumerate(folds, 1):
    print(f"\n    FOLD {fold_num}")
    print(f"Test indices: {[i + 1 for i in testFold]}")

    X_train = [X[j] for j in range(len(X)) if j not in testFold]
    y_train = [y[j] for j in range(len(y)) if j not in testFold]
    X_test = [X[j] for j in testFold]
    y_test = [y[j] for j in testFold]

    print(f"Training points: {[i + 1 for i in range(len(X)) if i not in testFold]}")
    print(f"Test points: {[i + 1 for i in testFold]}")

    for test_idx, (testPoint, actualLabel) in enumerate(zip(X_test, y_test)):
        print(f"\n  Testing point {testFold[test_idx] + 1}: {testPoint} (Actual: {actualLabel})")

        # Calculating and printing distances
        print("Distances to training points:")
        for i, trainPoint in enumerate(X_train):
            dist = manhattanDistance(trainPoint, testPoint)
            original_idx = [j for j in range(len(X)) if j not in testFold][i]
            print(f"    to Point {original_idx + 1}: {dist} (Label: {y_train[i]})")

        pred1 = predict(X_train, y_train, testPoint, 1)
        pred3 = predict(X_train, y_train, testPoint, 3)

        print(f"  k=1 Prediction: {pred1}")
        print(f"  k=3 Prediction: {pred3}")

        allPredictions1.append(pred1)
        allPredictions3.append(pred3)
        allActuals.append(actualLabel)

print("FINAL RESULTS")

print(f"Actual labels: {allActuals}")
print(f"Predictions for k=1: {allPredictions1}")
print(f"Predictions for k=3: {allPredictions3}")

# Calculating metrics manually
tp1, tn1, fp1, fn1, accuracy1, precision1, recall1, f1_1 = calculateMetrics(allActuals, allPredictions1)
tp3, tn3, fp3, fn3, accuracy3, precision3, recall3, f1_3 = calculateMetrics(allActuals, allPredictions3)

print("\nFor k=1:")
print(f"TP: {tp1}, TN: {tn1}, FP: {fp1}, FN: {fn1}")
print(f"Accuracy: {accuracy1:.4f}")
print(f"Precision: {precision1:.4f}")
print(f"Recall: {recall1:.4f}")
print(f"F1-measure: {f1_1:.4f}")

print("\nFor k=3:")
print(f"TP: {tp3}, TN: {tn3}, FP: {fp3}, FN: {fn3}")
print(f"Accuracy: {accuracy3:.4f}")
print(f"Precision: {precision3:.4f}")
print(f"Recall: {recall3:.4f}")
print(f"F1-measure: {f1_3:.4f}")

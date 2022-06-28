import numpy as np
import pathlib
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

path = str(pathlib.Path(__file__).parent.resolve())


# Get the data from files
def get_data(merge=False):
    # Read data
    print("Getting data...")
    data_time = time.time()

    train_data = np.genfromtxt(path + '\\data\\train_samples.txt', dtype='str', encoding='mbcs', delimiter='	', comments=None)
    train_labels = np.genfromtxt(path + '\\data\\train_labels.txt', delimiter='	', dtype='int')
    validation_data = np.genfromtxt(path + '\\data\\validation_samples.txt', dtype='str', encoding='mbcs', delimiter='	', comments=None)
    validation_labels = np.genfromtxt(path + '\\data\\validation_labels.txt', delimiter='	', dtype='int')
    test_data = np.loadtxt(path + '\\data\\test_samples.txt', dtype='str', encoding='mbcs', delimiter='	', comments=None)

    # Initial indexes
    test_index = [i[0] for i in test_data]
    train_index = [i[0] for i in train_data]
    validation_index = [i[0] for i in validation_data]

    # Data
    train_data = [i[1] for i in train_data]
    test_data = [i[1] for i in test_data]
    validation_data = [i[1] for i in validation_data]

    # Get labels
    train_labels = [i[1] for i in train_labels]
    validation_labels = [i[1] for i in validation_labels]

    # Take validation data into consideration
    if merge:
        train_data = np.append(train_data, validation_data)
        train_labels = np.append(train_labels, validation_labels)

    print(f"Getting data complete, it took: {time.time() - data_time} seconds.\n")

    return train_data, validation_data, test_data, train_index, validation_index, test_index, train_labels, validation_labels


# Generate the features from data
def get_features(train_data, test_data):
    # BagOfWords with countvectorizer
    print("Getting features...")
    features_time = time.time()

    # Initialize countvectorizer
    vectorizer = CountVectorizer(ngram_range=(3, 7),
                                 analyzer="char",
                                 strip_accents="unicode",
                                 binary=True,
                                 encoding="mbcs",
                                 max_features=1220000)

    # Create vocabulary
    vectorizer.fit(train_data)

    # Get features
    train_features = vectorizer.transform(train_data)
    test_features = vectorizer.transform(test_data)

    print(f"Getting features complete, it took: {time.time() - features_time} seconds.\n")

    return train_features, test_features


# Generate a prediction for the test data
def predict(train_features, test_features, train_labels):
    # Initializing the classifiers
    print("Predicting...")
    prediction_time = time.time()

    nb = MultinomialNB(alpha=0.1)
    rf = RandomForestClassifier()
    mlp = MLPClassifier(hidden_layer_sizes=10)


    # Predict using naive bayes
    nb_time = time.time()
    nb.fit(train_features, train_labels)
    predictionNB = nb.predict(test_features)
    print(f"Predicting using Naive Bayes took: {time.time() - nb_time} seconds.")

    # Predict using random forest
    rf_time = time.time()
    rf.fit(train_features, train_labels)
    predictionRF = rf.predict(test_features)
    print(f"Predicting using Random Forest took: {time.time() - rf_time} seconds.")

    # Predict using multi layer percepton
    mlp_time = time.time()
    mlp.fit(train_features, train_labels)
    predictionMLP = mlp.predict(test_features)
    print(f"Predicting using Multi-Layer Percepton took: {time.time() - mlp_time} seconds.")

    # Merge the predictions of all the algorithms
    prediction = []

    for i in range(len(predictionNB)):
        if predictionNB[i] == predictionRF[i] or predictionNB[i] == predictionMLP[i]:
            prediction.append(predictionNB[i])
        elif predictionRF[i] == predictionMLP[i]:
            prediction.append(predictionMLP[i])
        else:
            prediction.append(predictionNB[i])

    print(f"Predicting took: {time.time() - prediction_time} seconds.\n")

    return prediction


# Write in file a prediction
def write(prediction, test_index):
    # Writing in file
    print("Writing in file...")
    write_time = time.time()

    fout = open(path + "\\3AlgoOutput.csv", "w")
    fout.write("id,label\n")
    for i in range(len(prediction)):
        fout.write(f"{test_index[i]},{prediction[i]}\n")

    print(f"Writing in file complete, it took: {time.time() - write_time} seconds.\n")


# Generate statistics for prediction
def results(prediction, validation_labels):
    # Create confusion matrix and keep count of good guesses
    print("Generating results...")
    results_time = time.time()

    good = 0
    mat = np.zeros((3, 3))
    for i in range(len(validation_labels)):
        if validation_labels[i] == prediction[i]:
            good += 1
            mat[prediction[i] - 1][prediction[i] - 1] += 1
        else:
            mat[validation_labels[i] - 1][prediction[i] - 1] += 1

    # Print the data
    all = len(validation_labels)
    print(f"Good guesses: {str(good)} / {str(all)}")
    print(f"Accuracy: {str(good / all)}")
    print("Confusion matrix:")
    print(mat)
    print(f"Getting results complete, it took: {time.time() - results_time} seconds.\n")


# Predict test data
def predict_test_data():
    # Get data
    train_data, validation_data, test_data, train_index, validation_index, test_index, train_labels, validation_labels = get_data(merge=True)

    # Get the features
    train_features, test_features = get_features(train_data, test_data)

    # Train and predict
    prediction = predict(train_features, test_features, train_labels)

    # Write file
    write(prediction, test_index)


# Predict validation data
def predict_validation_data():
    # Get data
    train_data, validation_data, test_data, train_index, validation_index, test_index, train_labels, validation_labels = get_data(merge=False)

    # Get the features validation included
    train_features, validation_features = get_features(train_data, validation_data)

    # If we want to predict the validation features
    prediction = predict(train_features, validation_features, train_labels)

    # Print confusion matrix and other data for validation data
    results(prediction, validation_labels)


if __name__ == "__main__":
    start_time = time.time()

    # Get results for test data
    predict_test_data()

    # Get statistics for validation data
    #predict_validation_data()

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
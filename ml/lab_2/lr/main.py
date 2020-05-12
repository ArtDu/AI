# imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from model import LogisticRegressionUsingGD
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data(path):
    marks_df = pd.read_csv(path)
    return marks_df


if __name__ == "__main__":
    # load the data from the file
    data = load_data("../datasets/clean_tmdb.csv")

    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]

    # y = target values, last column of the data frame
    y = data.iloc[:, -1]


    # preparing the data for building the model

    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Logistic Regression from scratch using Gradient Descent
    model = LogisticRegressionUsingGD()
    model.fit(X_train, y_train, theta)
    predicted_classes = model.predict(X_test)
    accuracy = model.accuracy(predicted_classes, y_test.flatten())
    parameters = model.w_
    print("The accuracy of the model is {}".format(accuracy))
    print("The model parameters using Gradient descent")
    print("\n")
    print(parameters)

    # plotting the decision boundary
    # As there are two features
    # wo + w1x1 + w2x2 = 0
    # x2 = - (wo + w1x1)/(w2)

    # Using scikit-learn
    model = LogisticRegression()
    model.fit(X_train, y_train)
    parameters = model.coef_
    predicted_classes = model.predict(X_test)
    accuracy = accuracy_score(predicted_classes, y_test.flatten())
    print('The accuracy score using scikit-learn is {}'.format(accuracy))
    print("The model parameters using scikit learn")
    print(parameters)

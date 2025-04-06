import csv
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

def normalize_vector(vec):
    norm = math.sqrt(sum(x ** 2 for x in vec))
    return [x / norm for x in vec] if norm != 0 else vec

class Stats:
    @staticmethod
    def classification_report(y_true, y_pred):
        TP,FP,FN,TN = 0,0,0,0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i] == 0:
                TN += 1
            if y_true[i] == y_pred[i] == 1:
                TP += 1
            if y_true[i] == 1 != y_pred[i]:
                FN += 1
            if y_true[i] == 0 != y_pred[i]:
                FP += 1
        return TP,FP,FN,TN
    @staticmethod
    def accuracy(TP,FP,FN,TN):
        total = TP + FP + FN + TN
        if total == 0:
            return 0
        return (TP + TN) / total

    @staticmethod
    def precision(TP,FP):
        return TP/(TP+FP)

    @staticmethod
    def recall(TP,FN):
        return TP/(TP+FN)

    @staticmethod
    def f_score(y_true, y_pred):
        TP, FP, FN, TN = Stats.classification_report(y_true, y_pred)
        precision = Stats.precision(TP,FP)
        recall = Stats.recall(TP,FN)
        return 2 * precision * recall/(precision+recall)



class Embedding:
    def __init__(self, file_name):
        self. x = []
        self. y = []
        self.file_name = file_name
    def load_data_from_file(self):
        try:
            with open("data/" + self.file_name, newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    self.x.append(row[1])
                    self.y.append(row[0])
        except FileNotFoundError:
            raise Exception(f'file {self.file_name} not found')
    @staticmethod
    def vectorize(str):
        vec = [0.0 for _ in range(26)]
        for i in range(len(str)):
            letter = str[i]
            if ord('A') <= ord(letter) < ord('Z'):
                letter = letter.lower()
            letter = ord(letter) - ord('a')
            if 0 <= letter <= ord('z') - ord('a'):
                vec[letter] += 1
        return vec
    def vectorize_x(self):
        for i in range(len(self.x)):
            self.x[i] = normalize_vector(self.vectorize(self.x[i]))
            self.x[i] = np.array(self.x[i])

    def get_vectors(self):
        self.load_data_from_file()
        self.vectorize_x()
        return np.array(self.x), self.y


class Model:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        train = Embedding('lang.train.csv')
        test = Embedding('lang.test.csv')
        self.x_train, self.y_train = train.get_vectors()
        self.x_test, self.y_test = test.get_vectors()
        self.labels = {}
        for label in self.y_train:
            self.labels[label] = self.labels.get(label,0) + 1
        self.options = list(self.labels.keys())
        self.number_of_unique_labels = len(self.labels)
        self.weights = np.random.rand(self.number_of_unique_labels, len(self.x_train[0]))
        self.bias = np.random.rand(self.number_of_unique_labels)

    def calculate_correct(self, y_label):
        return self.options.index(y_label)

    def y_to_labels(self, y):
        y_data = y.copy()
        for i in range(len(y_data)):
            y_data[i] = self.calculate_correct(y_data[i])
        return y_data

    def predict(self, x_vector):
        return np.matmul(self.weights , x_vector.T) + self.bias

    def predict_all(self, x_data):
        predictions = np.empty(len(x_data))
        for i in range(len(x_data)):
            predictions[i] = np.argmax(self.predict(x_data[i]))
        return predictions

    @staticmethod
    def get_accuracy(predictions, y_data):
        correct_predictions = np.sum(predictions == y_data)
        accuracy = correct_predictions / len(y_data)
        return accuracy

    def update_perceptron_weights(self, perceptron_index,prediction,x_vec, d):
        y = prediction[perceptron_index]
        for i in range(len(x_vec)):
            self.weights[perceptron_index][i] = self.weights[perceptron_index][i] + self.learning_rate * (d - y) * x_vec[i]
        self.bias[perceptron_index] = self.bias[perceptron_index] + self.learning_rate * (d - y)

    def update_all_perceptrons(self, example_index):
        x_vec = self.x_train[example_index]
        prediction = self.predict(x_vec)
        for i in range(self.number_of_unique_labels):
            label = self.calculate_correct(self.y_train[example_index])
            if label == i:
                d = 1
            else:
                d = 0
            self.update_perceptron_weights(i,prediction,x_vec, d)

    def learn(self, epochs):
        accuracies = []
        for i in range(epochs):
            for j in range(len(self.x_train)):
                self.update_all_perceptrons(j)
            predictions = self.predict_all(self.x_train)
            train_accuracy = Model.get_accuracy(predictions,self.y_to_labels(self.y_train))
            test_accuracy = Model.get_accuracy(self.predict_all(self.x_test),self.y_to_labels(self.y_test))
            accuracies.append(train_accuracy)
            if i % 10 == 0:
                print(f'epoch: {i}')
                print(f'train accuracy: {train_accuracy}')
                print(f'test accuracy: {test_accuracy}')
        return accuracies

args = sys.argv
if len(args) != 3 and len(args) != 4:
    raise Exception("Wrong number of arguments")
learning_rate = float(args[1])
epochs = int(args[2])

m = Model(learning_rate)
accuracies = m.learn(epochs)

if len(args) == 4 and args[3] == "graph":
    plt.plot([i for i in range(epochs)], accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train accuracy after n Epochs")
    plt.show()
else:
    while (True):
        vec = input("input string to predict language: ")
        x = Embedding.vectorize(vec)
        p = m.predict(np.array(x))
        i = np.argmax(p)
        print(m.options[i])













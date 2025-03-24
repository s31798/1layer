import csv

def load_data_from_file(file_name):
    x = []
    y = []
    try:
        with open("data/" + file_name, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                vec = []
                text = row[1]
                for char in text:
                    vec.append(ord(char))
                x.append(vec)
                y.append(row[0])

        return x, y
    except FileNotFoundError:
        raise Exception(f'file {file_name} not found')

x,y = load_data_from_file("lang.train.csv")
print(x[0])




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

def precision(y_true, y_pred):
    TP, FP, FN, TN = classification_report(y_true, y_pred)
    return TP/(TP+FP)

def recall(y_true, y_pred):
    TP, FN, TN, FP = classification_report(y_true, y_pred)
    return TP/(TP+FN)

def f_score(y_true, y_pred):
    return 2*precision(y_true, y_pred)* recall(y_true, y_pred)/(precision(y_true, y_pred)+recall(y_true, y_pred))

y_true = [0,1,0,1,1,1,0]
y_pred = [0,1,1,1,1,0,0]
print(f_score(y_true, y_pred))
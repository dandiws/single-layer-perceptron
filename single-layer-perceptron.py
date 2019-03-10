import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

predict = lambda output: 0 if output<0.5 else 1
update_weight = lambda weight,alpha,delta: weight-alpha*delta
lostFn = lambda target,output: (output-target)**2

#output function
def outputFn(inputs, weights, bias):
    sum = bias
    for i in range(len(weights)):
        sum += inputs[i] * weights[i]
    return 1/(1+np.exp(-sum))

def delta_weight(target,output,inpoet=1):
    return 2*(output-target)*(1-output)*output*inpoet

def train_weights(train,weights,bias, alpha, n_epoch):    
    weights = weights[:]    
    errors=[]
    accuracy=[]
    for epoch in range(n_epoch):
        sum_error = 0.0        
        predictions = []
        targets = []
        for row in train:
            output = outputFn(row,weights,bias)

            #predict
            prediction = predict(output)
            predictions.append(prediction)
            targets.append(row[-1])

            #calculate error
            error = lostFn(row[-1],output)
            sum_error += error


            #update weight and bias
            bias = update_weight(bias,alpha,delta_weight(row[-1],output,1))    #new bias        
            for i in range(len(weights)):
                weights[i] = update_weight(weights[i],alpha,delta_weight(row[-1],output,row[i]))
        
        errors.append(sum_error)        
        accuracy.append(calculate_accuracy(targets,predictions))

    return (weights,bias,errors,accuracy)

# Split a dataset into n folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = np.random.randint(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def calculate_accuracy(targets,predictions):
    correct = 0
    n_data=len(targets)
    for i in range(n_data):
        if(targets[i]==predictions[i]): correct+=1
    #return in percentage (%)
    return correct/n_data*100.0

def validation(train, test, weights,bias, alpha, n_epoch):    
    weights,bias,train_errors,train_accuracy = train_weights(train,weights,bias, alpha, n_epoch)    
    test_errors = [] 
    test_accuracy = []
    for epoch in range(300):
        sum_error=0
        predictions = []
        targets = []
        for row in test:                
            output = outputFn(row,weights,bias)
            prediction = predict(output)  
            predictions.append(prediction)
            targets.append(row[-1])
            sum_error+= lostFn(row[-1],output)
        
        test_errors.append(sum_error)
        test_accuracy.append(calculate_accuracy(targets,predictions))    

    return (train_errors,test_errors,train_accuracy,test_accuracy)

def train_test_split(folds,test_index):
    n = len(folds)                
    train = list(folds)
    test = train.pop(test_index)
    test = list(test)  
    train = sum(train,[])  
    return (train,test)

def draw_graph(data_train,data_test,label,ylabel,title):
    plt.plot(data_train,label=label+' (Train)')
    plt.plot(data_test,label=label+' (Test)')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()    
    # plt.savefig(title+'.jpg')
    plt.show()    
    
def main(dataset, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)    
    train_errors_folds = []
    test_errors_folds = []
    train_accuracy_folds = []
    test_accuracy_folds = []

    for i in range(len(folds)):        
        train_set,test_set = train_test_split(folds,i)
        result = validation(train_set, test_set, *args)

        train_errors_folds.append(result[0])
        test_errors_folds.append(result[1])  
        train_accuracy_folds.append(result[2])
        test_accuracy_folds.append(result[3])              
    
    #draw average errors (train and test) for n fold
    draw_graph(np.mean(train_errors_folds,axis=0),
                np.mean(test_errors_folds,axis=0),
                'Average Error',
                'Error',
                'Grafik Error (alpha='+str(args[2])+')')   

    #draw average accuracy (train and test) for n fold
    draw_graph(np.mean(train_accuracy_folds,axis=0),
                np.mean(test_accuracy_folds,axis=0),
                'Average Accuracy',
                'Accuracy (%)',
                'Grafik akurasi (alpha='+str(args[2])+')')


DIR='D:\\Kuliah\\6 - Pembelajaran Mesin\\tugas3'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = pd.read_csv(DIR+'\\iris.csv',names=names)
training_data = iris.head(100).values.tolist()
for row in training_data:
    row[-1]=0 if row[-1]=="Iris-setosa" else 1

n_epoch = 300
k_fold = 5
weights_init = [0.5 for i in range(4)]
bias_init = 0.5

#using learning rate (alpha) = 0.2 and 0.8
main(training_data, k_fold,weights_init,bias_init, 0.2, n_epoch)
main(training_data, k_fold,weights_init,bias_init, 0.8, n_epoch)

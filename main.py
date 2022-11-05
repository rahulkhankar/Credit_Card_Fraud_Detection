import numpy as np
import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import csv


def Rand_Forest(x_train,x_test,y_train,y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    rf = RandomForestClassifier(criterion='gini')
    maxFeatures = range(1,x_train.shape[1]) #Maximum number of feature will be number of feature in one row
    param_dist = dict(max_features=maxFeatures) #Maximum features selection
    rand = RandomizedSearchCV(rf, param_dist, cv=10, scoring='accuracy', n_iter=len(maxFeatures))
    rand.fit(x_train,y_train)
    print(rand.best_estimator_)
    randomForest = RandomForestClassifier(bootstrap=True,criterion = "gini",max_features=rand.best_estimator_.max_features)
    randomForest.fit(x_train,y_train)
    prediction = randomForest.predict(x_test)    
    
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test,prediction)
    print("confusion matrix")
    print(confusion_matrix)
    from sklearn.metrics import classification_report
    print("classification report")
    print(classification_report(y_test, prediction))
    
    with open("Rforestprediction.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "prediction"])
        for i, p in enumerate(prediction):
            writer.writerow([i, p])

        
def MLNN(x_train,x_test,y_train,y_test):
    from sklearn.neural_network import MLPClassifier
    clf_NN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3,1))
    clf_NN.fit(x_train,y_train)     
    prediction = clf_NN.predict(x_test)
    
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test,prediction)
    print("confusion matrix")
    print(confusion_matrix)
    from sklearn.metrics import classification_report
    print("classification report")
    print(classification_report(y_test, prediction))
    
    with open("MLNNprediction.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "prediction"])
        for i, p in enumerate(prediction):
            writer.writerow([i, p])


def SVM(x_train,x_test,y_train,y_test):
    from sklearn import svm
    clf_svm=svm.SVC(kernel="poly")
    clf_svm.fit(x_train,y_train)    
    prediction=clf_svm.predict(x_test)
    
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test,prediction)
    print("confusion matrix")
    print(confusion_matrix)
    from sklearn.metrics import classification_report
    print("classification report")
    print(classification_report(y_test, prediction))
    
    with open("SVM.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "prediction"])
        for i, p in enumerate(prediction):
            writer.writerow([i, p])
    
def logtrain(x_train,x_test,y_train,y_test):
    from sklearn.linear_model import LogisticRegression
    logmodel=LogisticRegression()
    logmodel.fit(x_train,y_train)   
    prediction=logmodel.predict(x_test)
    
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test,prediction)
    print("confusion matrix")
    print(confusion_matrix)
    from sklearn.metrics import classification_report
    print("classification report")
    print(classification_report(y_test, prediction))
    
    with open("predictionsDefault.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "prediction"])
        for i, p in enumerate(prediction):
            writer.writerow([i, p])


def read_dataset():
    xtrain = []
    reader1=pd.read_csv('creditcard_train.csv')
    xtrain=pd.DataFrame(reader1)
    count_classes=pd.value_counts(xtrain['isFradulent'],sort=True)
    count_classes.plot(kind='bar' , rot=0)
    plt.title("transaction class distribution")
    plt.xticks(range(2))
    plt.xlabel("isFradulent")
    plt.ylabel("Frequency")
    print(xtrain.info())
    

    return xtrain

def preprocess_dataset(xdata):
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    
    #print(xdata.info())
    #Dropping the duplicate value from data set
    xdata=xdata.drop_duplicates()
    
    #Dropping column of Merchant_id and Transaction date
    xdata=xdata.drop(['Merchant_id','Transaction date'],axis=1)
    #print(xdata.info())
    
    
    #Label encoding
    encode=LabelEncoder()
    xdata['Is declined']=encode.fit_transform(xdata.values[:,2])
    xdata['isForeignTransaction']=encode.fit_transform(xdata.values[:,4])
    xdata['isHighRiskCountry']=encode.fit_transform(xdata.values[:,5])
    xdata['isFradulent']=encode.fit_transform(xdata.values[:,9])
    

    #One Hot Encoding  
    xtrain=np.array(xdata).astype(int)
    
    hotencode=OneHotEncoder(categorical_features=[2])
    xtrain=hotencode.fit_transform(xtrain).toarray()
    
    hotencode=OneHotEncoder(categorical_features=[5])
    xtrain=hotencode.fit_transform(xtrain).toarray()
    
    hotencode=OneHotEncoder(categorical_features=[7])
    xtrain=hotencode.fit_transform(xtrain).toarray()
    
    
    
    
    # Normalise
    xdata=pd.DataFrame(xtrain)

    
    y=xdata.iloc[:,[0,1,2,3,4,5]]
    x=xdata.iloc[:,[6,7,8,9,10,11]]
    ytrain=xdata.iloc[:,12]
    
    
    x=np.transpose(x)
    norm=preprocessing.Normalizer()
    x=norm.fit_transform(x)
    x=np.transpose(x)
    
    
    xtrain_processed = np.append(y, x, axis = 1)
    return np.array(xtrain_processed),np.array(ytrain)

def main():
    xtrain = read_dataset()
    xtrainprocessed,ytrainprocessed= preprocess_dataset(xtrain)
    
    unique,count=np.unique(ytrainprocessed,return_counts=True)
    ytrain_dict_value_count={k:v for (k,v) in zip(unique,count)}
    print(ytrain_dict_value_count)
    sm=SMOTE(random_state=0,ratio=1)
    sm_data_x,sm_data_y=sm.fit_sample(xtrainprocessed,ytrainprocessed)
    unique,count=np.unique(sm_data_y,return_counts=True)
    sm_data_y_dict_value_count={k:v for (k,v) in zip(unique,count)}
    print(sm_data_y_dict_value_count)
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(xtrainprocessed,ytrainprocessed,test_size=0.2)
    print("Result Before using oversampling")
    Rand_Forest(x_train,x_test,y_train,y_test)
    
    print("Result after oversampling")
    Rand_Forest(sm_data_x,x_test,sm_data_y,y_test)
    

    # ytest = model.predict(xtestprocessed)
if __name__ == '__main__':
    main()
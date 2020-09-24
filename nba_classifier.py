#import librarires 
import sklearn 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')
import pylabsklearn.metrics.accuracy_score
from sklearn import svm 
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans 

#read-in csv file (source: basketball reference)
nba_pos=pd.read_csv("nba_classify.csv",delimiter=",")

#drop columns (empty)
nba_pos=nba_pos.drop('Unnamed: 24',1)
nba_pos=nba_pos.drop('Unnamed: 19',1)
nba_pos=nba_pos.drop('Tm',1)

#fill in missing values (with median)
nba_pos=nba_pos.fillna(method='ffill')

#assign numbers for each basketball position
nba_pos.Pos=pd.Categorical(nba_pos.Pos)
nba_pos['pos_code']=nba_pos.Pos.cat.codes

#define the target variable
X=nba_pos[nba_pos.columns[3:26]]
X1=nba_pos.iloc[:,3:27,] 
y=nba_pos.pos_code

#train and test data sets 
X_train,X_test,y_train,y_test=train_test_split(nba_pos,y,
test_size=0.2)

#create a random forest object
model=RandomForestClassifier(n_estimators=1000,max_features=2,oob_score=True) 

#train the model using the train data set 
model.fit(X,y)

#Predict the output
predicted=model.predict(X)

#check accuracy
print(model.oob_score_)
#rcheck feature importance 
for X, imp in zip(X,model.feature_importances_):
    print(X,imp)
    
#SVM (support vector machine) classification
X=nba_pos.iloc[:,3:26] #select columns? (mas importante!)
y=nba_pos.pos_code

#Split the data into test and train 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Linear Kernel classification 
svc_linear=svm.SVC(kernel='linear',C=1)
svc_linear.fit(X_train,y_train)
prediction=svc_linear.predict(X_test)
confusion_mat=confusion_matrix(y_test,prediction)
print(confusion_mat)
accuracy_score(y_test,prediction)

# statistics by position (pivot table)
pos_stats=pd.pivot_table(nba_pos,index=['Pos'],aggfunc='mean')
# k-means algorithm to group players

#create a kmeans model using 5 clusters 
k_means_model=KMeans(n_clusters=5,random_state=None).fit(nba_pos.iloc[:,3:19])
labels=k_means_model.labels_
print(pd.crosstab(labels,nba_pos["Pos"]))

#misclassified basketball positions 
pg_outside=nba_pos[(labels==2) & (nba_pos["Pos"]=="PG")] #point guards that match up to centers? 

#data visualizations
#a.TRB% histogram 
nba_reb=nba_pos["TRB%"]
reb=nba_reb.plot.hist(alpha=0.5)
reb.set_ylabel("Frequency")
reb.set_xlabel("TRB%")
pylab.show()

#b. TRB% by position 
nba_pos["TRB%"].hist(by=nba_pos['Pos'])
pylab.show() 

#c. OWS and DWS by position
nba_pos["OWS"].hist(by=nba_pos['Pos'])
pylab.show() 

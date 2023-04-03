# -*- coding: utf-8 -*-
"""
Created on Sat Mar 5 12:59:32 2022

@author: Rahul
"""

# Regression Template

# Random Forest Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import VarianceThreshold 
from sklearn.feature_selection import RFE
#from statistics import mean
#from statistics import stdev
# Importing the datasets

datasets = pd.read_csv('GBA_gcn.csv')
X = datasets.iloc[:, 5:24].values
y = datasets.iloc[:, 24].values
#print(datasets.corr())
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#print(X_train.shape)

#standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X.fit(X)
X_train_sc = sc_X.transform(X_train)
X_test_sc = sc_X.transform(X_test)
#Feature selection
vt = VarianceThreshold(threshold=(0.8*(1-0.8)))
vt.fit(X)
X_train = vt.transform(X_train)
X_test = vt.transform(X_test)
X_train_sc = vt.transform(X_train_sc)
X_test_sc = vt.transform(X_test_sc)
X = vt.transform(X)
#print(X_train.shape)
#print(X_train_sc.shape)
###################################################################################
#                        Correlation                                              #
###################################################################################
from matplotlib import cm as cm
import seaborn as sns   ## conda install seaborn
##
##      Covariance matrix
##
def correl_matrix(X,cols):
    fig = plt.figure(figsize=(23,23), dpi=100)
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet',30)
    cax = ax1.imshow(np.abs(X.corr()),interpolation='nearest',cmap=cmap)
##    ax1.set_xticks(major_ticks)
    major_ticks = np.arange(0,len(cols),1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True,which='both',axis='both')
##    plt.aspect('equal')
    plt.title('Correlation Matrix')
    labels = cols
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=12)
    fig.colorbar(cax, ticks=[-0.4,-0.25,-.1,0,0.1,.25,.5,.75,1])
    plt.show()
    return(1)
##
##      make pair plots
##
def pairplotting(df):
    sns.set(style='whitegrid', context='notebook')
    cols = df.columns[2:]
  
##bcols = ['age', 'sex', 'cpt', 'rbp', 'sc', 'fbs', 'rer', 'mhr', 'eia', 'opst', 'dests', 'nmvcf', 'thal', 'a1p2']
    sns.pairplot(df[cols],height=10)
    plt.show()
##
##      read csv
##   
import pandas as pd
#df = pd.read_csv('GBA.csv')
ypd = datasets['PBA'].values
Size = len(ypd)
Xpd = np.hstack((datasets.iloc[:,2].values.reshape(Size,1),datasets.iloc[:,3:].values))
#X = df.iloc[:,2:4].values      ##  not including Year
cols = datasets.columns

#print(cols[24])
##
##      correlation matrix
##
#print(' Covariance Matrix ')
correl_matrix(datasets.iloc[:,2:],cols[2:])
##
##      Pair plotting
##
#print(' Pair plotting ')
#pairplotting(df)
##


################################Linear Regression###########################################
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train_sc, y_train)
Y_Pred_lin_t = linreg.predict(X_train_sc)
Y_Pred_lin = linreg.predict(X_test_sc)
print('***********Linear Regression******************')
loss_i = 0
for i in range(0,len(Y_Pred_lin)):
    loss_i = loss_i+np.sqrt(((y_test[i] - Y_Pred_lin[i])**2))


loss_k = 0
for i in range(0,len(y_train)):
    loss_k = loss_k +np.sqrt(((y_train[i] - Y_Pred_lin_t[i])**2))
lin_loss = loss_i/len(Y_Pred_lin)
print('Average error train for Linear Regression',loss_k/len(Y_Pred_lin_t))    
print('Average error test for Linear Regression',loss_i/len(Y_Pred_lin))
loss_lin = np.sqrt(((y_test - Y_Pred_lin)**2))
#Feature selection based on Linear Regression Model
selector = RFE(linreg, n_features_to_select=10, step=1)
selector.fit(X, y)
X_test_sc = selector.transform(X_test_sc)
X_train_sc = selector.transform(X_train_sc)
#X_test = selector.transform(X_test)
#X_train = selector.transform(X_train)
#print(X_train_sc.shape)
#print(X_test_sc.shape)
#print(X_train.shape)
#print(X_test.shape)
#print('Feature Selection', selector.support_)

############################## Plots #######################################
plt.ylim([0,1600])
plt.xlim([2000,2100])
plt.plot(y_test,color='red',label="Real PBA")
plt.plot(Y_Pred_lin,color='blue',label="Predicted PBA")
plt.title("GBA-PBA VS Acutal PBA Linear Regression")
plt.xlabel("paths") 
plt.ylabel("delay")
plt.legend()
plt.figure(figsize=(50,8))
plt.show()
plt.ylim([-300,350])
plt.xlim([2000,2100])
plt.plot(loss_lin, color= 'yellow', label= "loss")
plt.title("Linear Regression LOSS")
plt.xlabel("Paths") 
plt.ylabel("Loss")
plt.legend()
plt.figure(figsize=(500,8))
plt.show()
#############################################################################

################################SVM###########################################
from sklearn.svm import SVR
SVM_Model = SVR(degree=20).fit(X_train_sc, y_train)


print('***************SVM****************************')
Y_Pred_SVM = SVM_Model.predict(X_test_sc)
Y_Pred_SVM_t = SVM_Model.predict(X_train_sc)
loss_i = 0
for i in range(0,len(Y_Pred_SVM)):
    loss_i = loss_i+np.sqrt(((y_test[i] - Y_Pred_SVM[i])**2))

svm_loss = loss_i/len(Y_Pred_SVM)
loss_k = 0
for i in range(0,len(y_train)):
    loss_k = loss_k +np.sqrt(((y_train[i] - Y_Pred_SVM_t[i])**2))
print('Average error train',loss_k/len(Y_Pred_SVM_t))    
print('Average error test',loss_i/len(Y_Pred_SVM))
loss_svm = np.sqrt(((y_test - Y_Pred_SVM)**2))

############################## Plots #######################################
plt.ylim([0,1600])
plt.xlim([2000,2100])
plt.plot(y_test,color='red',label="Real PBA")
plt.plot(Y_Pred_SVM,color='blue',label="Predicted PBA")
plt.title("GBA-PBA VS Acutal PBA SVM")
plt.xlabel("paths") 
plt.ylabel("delay")
plt.legend()
plt.figure(figsize=(50,8))
plt.show()
plt.ylim([-300,350])
plt.xlim([2000,2100])
plt.plot(loss_svm, color= 'yellow', label= "loss")
plt.title("SVM LOSS")
plt.xlabel("Paths") 
plt.ylabel("Loss")
plt.legend()
plt.figure(figsize=(500,8))
plt.show()


##############################Random Forest Regressor##########################
print("***********Random Forest Regressor*************")
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 200, random_state = 1,max_depth= 6, max_features=0.6, max_samples=6750,oob_score= True )
regressor.fit(X_train,y_train)


"""
X_selec = np.zeros(np.shape(X))
p=0
for l in range(0,len(selector.support_)):
    if(selector.support_[l]):
        X_selec[:,p] = X[:,l]
        print('hi')
    p = p+1
print(X_selec)
"""
# Predicting a new result with the Random Forest Regression

Y_Pred = regressor.predict(X_test)
Y_Pred_t = regressor.predict(X_train)
#Loss Calculation

loss_j = 0
for i in range(0,len(Y_Pred)):
    loss_j = loss_j +np.sqrt(((y_test[i] - Y_Pred[i])**2))
loss_k = 0


loss_RF = loss_j/len(Y_Pred)



for i in range(0,len(y_train)):
    loss_k = loss_k +np.sqrt(((y_train[i] - Y_Pred_t[i])**2))    
print('Train accuracy \n Average PBA vs GBA-PBA',loss_k/len(Y_Pred_t))
#print(y_test - Y_Pred)
#Accuracy calculation
print('Test accuracy \n Average PBA vs GBA-PBA',loss_j/len(Y_Pred))
#print(X_test[0][21])
#from sklearn.metrics import accuracy_score
#print('Accuracy: test %.2f' % accuracy_score(Y_Pred, y_test))

loss = np.sqrt(((y_test - Y_Pred)**2))/20
#print('LOSS',loss)


############################## Plots #######################################
plt.ylim([0,1600])
plt.xlim([2000,2100])
plt.plot(y_test,color='red',label="Real PBA")
plt.plot(Y_Pred,color='blue',label="Predicted PBA")
plt.title("GBA-PBA VS Acutal PBA Random Forest")
plt.xlabel("paths") 
plt.ylabel("delay")
plt.legend()
plt.figure(figsize=(50,8))
plt.show()

plt.ylim([-40,70])
plt.xlim([2000,2100])
plt.plot(loss, color= 'yellow', label= "loss")
plt.title("Random Forest Loss")
plt.xlabel("Paths") 
plt.ylabel("Loss")
plt.legend()
plt.figure(figsize=(50,8))
plt.show()


#######################XGBRandomForest#########################################
print("***********GRADIENT BOOSTED RANDOM FOREST REGRESSION**********")
from sklearn.ensemble import GradientBoostingRegressor
xgbrf = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=1, loss='ls', subsample = 0.6, tol = 1e-4, criterion= 'mse')
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(xgbrf, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# report performance
xgbrf.fit(X_train,y_train)
Y_pred_x = xgbrf.predict(X_train)
Y_pred_xt = xgbrf.predict(X_test)

#Loss Calculation

loss_j = 0
for i in range(0,len(Y_pred_xt)):
    loss_j = loss_j +np.sqrt(((y_test[i] - Y_pred_xt[i])**2))
loss_k = 0
loss_gb = loss_j/(len(Y_pred_xt))
for i in range(0,len(y_train)):
    loss_k = loss_k +np.sqrt(((y_train[i] - Y_pred_x[i])**2))    
print('Train accuracy \n Average PBA vs GBA-PBA',loss_k/(len(Y_pred_x)))
print('Test accuracy \n Average PBA vs GBA-PBA',loss_j/(len(Y_pred_xt)))
loss = np.sqrt(((y_test - Y_Pred)**2))/2
############################## Plots #######################################
plt.ylim([0,1600])
plt.xlim([2000,2100])
plt.plot(y_test,color='red',label="Real PBA")
plt.plot(Y_pred_xt,color='blue',label="Predicted PBA")
plt.title("GBA-PBA VS Acutal PBA Gradient Boosted Decision Trees")
plt.xlabel("paths") 
plt.ylabel("delay")
plt.legend()
plt.figure(figsize=(50,8))
plt.show()

plt.ylim([-40,70])
plt.xlim([2000,2100])
plt.plot(loss, color= 'yellow', label= "loss")
plt.title("Gradient Boosted Random Forest Loss")
plt.xlabel("Paths") 
plt.ylabel("Loss")
plt.legend()
plt.figure(figsize=(50,8))
plt.show()
################################Table##################################
print("______________________________________________________________")
print("|-------------------Test Results of models-------------------|")
print('|Loss in Linear Regression              |',lin_loss,' |' )
print("----------------------------------------|--------------------|")
print('|Loss in SVM                            |',svm_loss ,'|')
print("----------------------------------------|--------------------|")
print('|Loss in Random Forest                  |',loss_RF,' |' )
print("----------------------------------------|--------------------|")
print('|Loss in Gradient Boosted Decision Trees|',loss_gb,'|' )
print("|____________________________________________________________|")
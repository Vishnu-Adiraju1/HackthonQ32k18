import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
 
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


diabetesDF = pd.read_csv('C:\\Users\\RA055337\\Desktop\\AI\\problem4_diabetes.csv')

diabetesDF['BMI'].replace(0, np.nan, inplace= True)
diabetesDF['Glucose'].replace(0, np.nan, inplace= True)
diabetesDF['SkinThickness'].replace(0, np.nan, inplace= True)

diabetesDF.dropna(inplace=True)
diabetesDF.drop_duplicates(inplace=True)

print(diabetesDF.shape);


diabetesDF.info() # output shown below

corr = diabetesDF.corr()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
 
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


diabetesDF = pd.read_csv('C:\\Users\\RA055337\\Desktop\\AI\\problem4_diabetes.csv')

print(diabetesDF.head());

print(diabetesDF.shape);

#diabetesDF=diabetesDF.sort_values(by=['Glucose'])

#diabetesDF.info() # output shown below
corr = diabetesDF.corr()

#print(corr)
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

diabetesDF['BMI'].replace(0, np.nan, inplace= True)
diabetesDF['Glucose'].replace(0, np.nan, inplace= True)
diabetesDF['SkinThickness'].replace(0, np.nan, inplace= True)


#diabetesDF['Insulin'].replace(0, np.nan, inplace= True)

diabetesDF.dropna(inplace=True)
diabetesDF.drop_duplicates(inplace=True)


		
dfTrain = diabetesDF[:373]
dfTest = diabetesDF[373:] 

#dfTrain = diabetesDF[:455]
#dfTest = diabetesDF[455:650] 

trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome',1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome',1))

means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
 
trainData = (trainData - means)/stds
testData = (testData - means)/stds
 
# np.mean(trainData, axis=0) => check that new means equal 0
# np.std(trainData, axis=0) => check that new stds equal 1

diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)

accuracy = diabetesCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")
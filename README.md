# EX-06 FEATURE TRANSFORMATION
## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## Algorithm:
### Step1:
Read the given Data.
### Step2: 
Clean the Data Set using Data Cleaning Process.
### Step3:
Apply Feature Transformation techniques to all the features of the data set.
### Step4:
Print the transformed features.
## Program:
Developed By: Naveenaa A K

Register No: 212222230094

### Importing libraries and reading csv file:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
df=pd.read_csv("Data_to_Transform.csv")
```
### Basic Information:
```
df.head()
df.info()
df.info()
```
![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/72d2d78c-0fb1-49f5-a07f-333e66ed5129)
  
### Before Transformation:
```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/5e82854d-26b6-40c1-978b-32e7d0602353)

```
sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.title("Highly Negative Skew")
plt.show()
```
![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/3979bffa-21ba-4627-8a87-eeb9e3c4bcfc)

```
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/ebb2084a-d071-4b7a-bc31-7aef5c457f7f)

```
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
 ![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/3d768a5d-b0ed-4a76-a927-ab14cc662469)

### Log Transformation:
```
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/5c07c1ea-c813-4aaa-8a44-0948726d002e)

```
df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/08656ed3-4840-4db2-bbc7-bc11ac494364)

### Reciprocal Transformation:
```
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/795fa4fc-60f2-4fdb-8cf6-e16beeff330a)

### SquareRoot Transformation:
```
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/62d30af5-5e7d-48ce-85a0-38da9ee46ff4)

### Power Transformation:
```
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/5082edbc-ab56-4d2b-98e5-f3097fdf3f9d)

```
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/43bb56f7-7bc7-46b7-acc9-943b8dd72000)
 
### Quantile Transformation:
```
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate  Negative Skew")
plt.show()
```
![image](https://github.com/naveenaakumarasamy/Datascience-Ex06/assets/113497406/dea278eb-c146-4201-bcf2-c479949d597e)

## Result:
Thus feature transformation is done for the given dataset.

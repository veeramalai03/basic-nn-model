# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
# Developed By: VEERAMALAI S
# Register Number:212220230056
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_csv("exp1.csv")
df.head()
x=df[['input']].values
x
y=df[['output']].values
y
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=40)

scaler=MinMaxScaler()
scaler.fit(xtrain)
scaler.fit(xtest)
xtrain1=scaler.transform(xtrain)
xtest1=scaler.transform(xtest)

model=Sequential([
    Dense(12,activation='relu'),
    Dense(8,activation='relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(xtrain1,ytrain,epochs=2000)
lossmodel=pd.DataFrame(model.history.history)
lossmodel.plot()
model.evaluate(xtest1,ytest)

xn1=[[20]]
xn11=scaler.transform(xn1)
model.predict(xn11)
```
## Dataset Information

![Screenshot1](https://user-images.githubusercontent.com/75234790/187118349-3cda2996-c400-40b4-86a3-666241b957a9.png)


## OUTPUT 

### Test Data Root Mean Squared Error
### New Sample Data Prediction
![Screenshot 2022-08-29 085511](https://user-images.githubusercontent.com/75234790/187118859-c54ec3e6-b7cd-40b3-a8f3-4cff2ec030c7.png)

![Screenshot 2022-08-29 085759](https://user-images.githubusercontent.com/75234790/187118864-de4d450b-5bf5-472d-af22-e764957d6987.png)



### Training Loss Vs Iteration Plot
![Screenshot 2022-08-29 085050](https://user-images.githubusercontent.com/75234790/187118881-53e6af26-c2f5-4b29-9e4d-7cb93d889383.png)


## RESULT
Thus,the neural network regression model for the given dataset is developed.

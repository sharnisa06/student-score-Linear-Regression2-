import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data={'hours':[1,2,3,4,5,6,7,8,9,10],'scores':[10,20,30,40,50,60,70,80,90,100]}
df=pd.DataFrame(data)

x=df[['hours']]
y=df[['scores']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("coefficient",model.coef_)
print("intercept",model.intercept_)

plt.scatter(x,y,color="blue")
plt.plot(x,model.predict(x),color="red")
plt.xlabel("hours studied")
plt.ylabel("scores")
plt.show()
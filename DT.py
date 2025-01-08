import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import Bunch
import pandas as pd
from sklearn.metrics import r2_score
def MAPE(true, pred):
  diff = np.abs(np.array(true) - np.array(pred))
  return np.mean(diff / true)

df = pd.read_excel('20240704DATASETnew.xlsx')
data_column=['mc','$K_s$']
data_feature=['mc','$K_s$']
bunch = Bunch(
  data=df.drop(data_feature,axis=1).values,
  target=df['$K_s$'].values,
  feature_names=df.drop(data_feature, axis=1).columns.tolist(),
  target_names=['$K_s$']
)
X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
y = bunch.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model=DecisionTreeRegressor(random_state=123,max_depth=13,min_samples_leaf=1,max_leaf_nodes=69)
model.fit(X_train,y_train)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
sorted_index=model.feature_importances_.argsort()
X=pd.DataFrame(bunch.data,columns=bunch.feature_names)
plt.figure(figsize=(9,6))
import matplotlib
matplotlib.rcParams['font.family']='Times New Roman'
matplotlib.rcParams['font.size']='12'
plt.barh(range(X.shape[1]),
         model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]),X.columns[sorted_index])
plt.xlabel('Feature Importance',fontdict={'family': 'Times New Roman','size':12})
plt.ylabel('Feature',fontdict={'family': 'Times New Roman','size':12})
plt.title('Decision Tree',fontdict={'family': 'Times New Roman','size':12})
plt.tight_layout()
plt.show()
RMSE1=[]
RMSE2=[]
RMSE3=[]
RMSE4=[]
for max_leaf_nodess in range(2,100):
  model1=DecisionTreeRegressor(random_state=123,max_leaf_nodes=max_leaf_nodess,min_samples_leaf=1)
  model1.fit(X_train,y_train)
  pred_oob=model1.score(X_test,y_test)
  RMSE1.append(pred_oob)
for max_leaf_nodess in range(2,100):
  model2=DecisionTreeRegressor(random_state=123,max_leaf_nodes=max_leaf_nodess,min_samples_leaf=2)
  model2.fit(X_train,y_train)
  pred_oob=model2.score(X_test,y_test)
  RMSE2.append(pred_oob)
for max_leaf_nodess in range(2,100):
  model3=DecisionTreeRegressor(random_state=123,max_leaf_nodes=max_leaf_nodess,min_samples_leaf=3)
  model3.fit(X_train,y_train)
  pred_oob=model3.score(X_test,y_test)
  RMSE3.append(pred_oob)
for max_leaf_nodess in range(2,100):
  model4=DecisionTreeRegressor(random_state=123,max_leaf_nodes=max_leaf_nodess,min_samples_leaf=4)
  model4.fit(X_train,y_train)
  pred_oob=model4.score(X_test,y_test)
  RMSE4.append(pred_oob)
plt.rcParams['font.family'] = 'Times New Roman'
plt.xticks(range(0, 101, 20))
plt.xlim(0, 100)
plt.yticks([i / 100 for i in range(-20, 101, 20)])
plt.ylim(-0.2, 1)
plt.plot(range(2,100),RMSE1,color='blue',label='min_samples_leaf=1')
plt.plot(range(2,100),RMSE2,color='black',label='min_samples_leaf=2')
plt.plot(range(2,100),RMSE3,color='red',label='min_samples_leaf=3')
plt.plot(range(2,100),RMSE4,color='green',label='min_samples_leaf=4')
plt.xlabel('Number of max_leaf_nodes',fontdict={'family': 'Times New Roman','size':14})
plt.ylabel(r'$R^2$',fontdict={'family': 'Times New Roman','size':14})
plt.title('DT',fontdict={'family': 'Times New Roman','size':14})
plt.legend()
plt.savefig('1DT.jpg')
Z7 = model.predict(X_train)
Z8 = model.predict(X_test)
xx = np.linspace(-1, 120, 1000)
yy = xx
sns.set_style("whitegrid")
sns.set(style="ticks")
plt.figure(figsize=(8,6))
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(xx, yy, '--',c='grey', linewidth=1.5)
plt.scatter(y_train, Z7,marker='o')
plt.scatter(y_test, Z8,c='darkorange',marker='s')

plt.tick_params (axis='both',labelsize=14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)

font1 = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 14,}
plt.axis('tight')
plt.xlabel('Tested Ks(kN/mm)',fontdict={'family': 'Times New Roman','size':14})
plt.ylabel('Predicted Ks(kN/mm)',fontdict={'family': 'Times New Roman','size':14})
plt.xlim([-1,120])
plt.ylim([-1,120])
plt.title('DT',fontdict={'family': 'Times New Roman','size':14})
plt.legend(['y = x','Training set','Testing set'], loc = 'upper left', prop={'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 14,})
plt.savefig('30数据决策树预测图.jpg')
plt.show()
Z7 = model.predict(X_train)
Z8 = model.predict(X_test)
print("DT Training R2:", r2_score(y_train, Z7), "RMSE:", np.sqrt(mean_squared_error(y_train, Z7)), "MAE:", mean_absolute_error(y_train, Z7),"MAPE:", MAPE(y_train, Z7))
print("DT Testing R2:", r2_score(y_test, Z8), "RMSE:", np.sqrt(mean_squared_error(y_test, Z8)), "MAE:", mean_absolute_error(y_test, Z8),"MAPE:", MAPE(y_test, Z8))

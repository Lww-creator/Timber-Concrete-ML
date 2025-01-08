import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import Bunch
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
adaboost_reg = AdaBoostRegressor( n_estimators=21, random_state=123,learning_rate=0.3230)
adaboost_reg.fit(X_train, y_train)
y_pred = adaboost_reg.predict(X_test)
Z7 = adaboost_reg.predict(X_train)
Z8 = adaboost_reg.predict(X_test)
print("AdaBoost Training R2:", r2_score(y_train, Z7), "RMSE:", np.sqrt(mean_squared_error(y_train, Z7)), "MAE:", mean_absolute_error(y_train, Z7),"MAPE:", MAPE(y_train, Z7))
print("AdaBoost Testing R2:", r2_score(y_test, Z8), "RMSE:", np.sqrt(mean_squared_error(y_test, Z8)), "MAE:", mean_absolute_error(y_test, Z8),"MAPE:", MAPE(y_test, Z8))
RMSE1=[]
RMSE2=[]
RMSE3=[]
RMSE4=[]
for n_estimators in range(2,200):
  adaboost_reg = AdaBoostRegressor( n_estimators=n_estimators, random_state=123,learning_rate=0.01)
  adaboost_reg.fit(X_train,y_train)
  pred_oob=adaboost_reg.score(X_test,y_test)
  RMSE1.append(pred_oob)
for n_estimators in range(2,200):
  adaboost_reg = AdaBoostRegressor( n_estimators=n_estimators, random_state=123,learning_rate=0.1)
  adaboost_reg.fit(X_train,y_train)
  pred_oob=adaboost_reg.score(X_test,y_test)
  RMSE2.append(pred_oob)
for n_estimators in range(2,200):
  adaboost_reg = AdaBoostRegressor( n_estimators=n_estimators, random_state=123,learning_rate=0.2)
  adaboost_reg.fit(X_train,y_train)
  pred_oob=adaboost_reg.score(X_test,y_test)
  RMSE3.append(pred_oob)
for n_estimators in range(2,200):
  adaboost_reg = AdaBoostRegressor( n_estimators=n_estimators, random_state=123,learning_rate=0.3)
  adaboost_reg.fit(X_train,y_train)
  pred_oob=adaboost_reg.score(X_test,y_test)
  RMSE4.append(pred_oob)
plt.rcParams['font.family'] = 'Times New Roman'
plt.xticks(range(0, 101, 20))
plt.xlim(0, 100)
plt.yticks([i / 100 for i in range(-80, 101, 30)])
plt.ylim(-0.8, 1)
plt.plot(range(2,200),RMSE1,color='blue',label='learning_rate=0.01')
plt.plot(range(2,200),RMSE2,color='black',label='learning_rate=0.1')
plt.plot(range(2,200),RMSE3,color='red',label='learning_rate=0.2')
plt.plot(range(2,200),RMSE4,color='green',label='learning_rate=0.3')
plt.xlabel('Number of n_estimators',fontdict={'family': 'Times New Roman','size':14})
plt.ylabel(r'$R^2$',fontdict={'family': 'Times New Roman','size':14})
plt.title('AdaBoost',fontdict={'family': 'Times New Roman','size':14})
plt.legend()
plt.savefig('AdaBoost1.jpg')
plt.rcParams['font.family'] = 'Times New Roman'
Z7 = adaboost_reg.predict(X_train)
Z8 = adaboost_reg.predict(X_test)
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
plt.title('AdaBoost',fontdict={'family': 'Times New Roman','size':14})
plt.legend(['y = x','Training set','Testing set'], loc = 'upper left', prop={'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 14,})
plt.savefig('AdaBooost2.jpg')
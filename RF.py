def MAPE(true, pred):
  diff = np.abs(np.array(true) - np.array(pred))
  return np.mean(diff / true)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import Bunch
import seaborn as sns
# 加载桌面上的 Excel 文件
df = pd.read_excel('20240704DATASETnew.xlsx')
data_column=['mc','$K_s$']
data_feature=['mc','$K_s$']
bunch = Bunch(
  data=df.drop(data_feature,axis=1).values,
  target=df['$K_s$'].values,
  feature_names=df.drop(data_feature, axis=1).columns.tolist(),
  target_names=['$K_s$']
)
X=pd.DataFrame(bunch.data,columns=bunch.feature_names)
y=bunch.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(random_state=123,n_estimators=18,max_depth=11,max_features=1,min_samples_leaf=1)
model.fit(X_train,y_train)
model.fit(X_train,y_train)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
Z7 = model.predict(X_train)
Z8 = model.predict(X_test)
print("RF Training R2:", r2_score(y_train, Z7), "RMSE:", np.sqrt(mean_squared_error(y_train, Z7)), "MAE:", mean_absolute_error(y_train, Z7),"MAPE:", MAPE(y_train, Z7))
print("RF Testing R2:", r2_score(y_test, Z8), "RMSE:", np.sqrt(mean_squared_error(y_test, Z8)), "MAE:", mean_absolute_error(y_test, Z8),"MAPE:", MAPE(y_test, Z8))
plt.figure(figsize=(9,6))
sorted_index=model.feature_importances_.argsort()
plt.barh(range(X.shape[1]),
         model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]),X.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest')
plt.tight_layout()
plt.show()
plt.savefig('随机森林')
RMSE1=[]
RMSE2=[]
RMSE3=[]
RMSE4=[]
for n_estimators in range(1,200):
  model1=RandomForestRegressor(random_state=0,min_samples_leaf=1,n_estimators=n_estimators)
  model1.fit(X_train,y_train)
  pred_oob=model1.score(X_test,y_test)
  RMSE1.append(pred_oob)
for n_estimators in range(1,200):
  model2=RandomForestRegressor(random_state=0,min_samples_leaf=2,n_estimators=n_estimators)
  model2.fit(X_train,y_train)
  pred_oob=model2.score(X_test,y_test)
  RMSE2.append(pred_oob)
for n_estimators in range(1,200):
  model3=RandomForestRegressor(random_state=0,n_estimators=n_estimators,min_samples_leaf=3)
  model3.fit(X_train,y_train)
  pred_oob=model3.score(X_test,y_test)
  RMSE3.append(pred_oob)
for n_estimators in range(1,200):
  model4=RandomForestRegressor(random_state=0,n_estimators=n_estimators,min_samples_leaf=4)
  model4.fit(X_train,y_train)
  pred_oob=model4.score(X_test,y_test)
  RMSE4.append(pred_oob)

plt.plot(range(1,200),RMSE1,color='blue',label='min_samples_leaf=1')
plt.plot(range(1,200),RMSE2,color='black',label='min_samples_leaf=2')
plt.plot(range(1,200),RMSE3,color='red',label='min_samples_leaf=3')
plt.plot(range(1,200),RMSE4,color='green',label='min_samples_leaf=4')
plt.xticks(range(0, 201, 40))
plt.xlim(0, 200)
plt.yticks([i / 100 for i in range(-80, 101, 30)])
plt.ylim(-0.8, 1)
plt.xlabel('Number of n_estimators',fontdict={'family': 'Times New Roman','size':16})
plt.ylabel(r'$R^2$',fontdict={'family': 'Times New Roman','size':16})
plt.title('Random Forest',fontdict={'family': 'Times New Roman','size':16})
plt.legend()
plt.savefig('RT参数.jpg')
plt.show()
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

plt.tick_params (axis='both',labelsize=16)
plt.yticks(fontproperties = 'Times New Roman', size = 16)
plt.xticks(fontproperties = 'Times New Roman', size = 16)

font1 = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 16,}
plt.axis('tight')
plt.xlabel('Tested Ks(kN/mm)',fontdict={'family': 'Times New Roman','size':16})
plt.ylabel('Predicted Ks(kN/mm)',fontdict={'family': 'Times New Roman','size':16})
plt.title('Random Forest',fontdict={'family': 'Times New Roman','size':16})
plt.xlim([-1,120])
plt.ylim([-1,120])
plt.legend(['y = x','Training set','Testing set'], loc = 'upper left', prop={'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 14,})
plt.savefig('RT.jpg')
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import Bunch
import seaborn as sns
def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)
df = pd.read_excel('20240704DATASETnew-NOtitle.xlsx')
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
model=GradientBoostingRegressor(random_state=123,n_estimators=63,max_depth=5,learning_rate=0.32095,min_samples_split=2)
model.fit(X_train,y_train)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
RMSE1=[]
RMSE2=[]
RMSE3=[]
RMSE4=[]
divisions = np.linspace(0.001, 1, 100)
for n_estimators in range(1,100):
  model1=GradientBoostingRegressor(random_state=123,n_estimators=n_estimators,learning_rate=0.01)
  model1.fit(X_train,y_train)
  pred_oob=model1.score(X_test,y_test)
  RMSE1.append(pred_oob)
for n_estimators in range(1,100):
  model2=GradientBoostingRegressor(random_state=123,n_estimators=n_estimators,learning_rate=0.05)
  model2.fit(X_train,y_train)
  pred_oob=model2.score(X_test,y_test)
  RMSE2.append(pred_oob)
for n_estimators in range(1,100):
  model3=GradientBoostingRegressor(random_state=123,n_estimators=n_estimators,learning_rate=0.1)
  model3.fit(X_train,y_train)
  pred_oob=model3.score(X_test,y_test)
  RMSE3.append(pred_oob)
for n_estimators in range(1,100):
  model4=GradientBoostingRegressor(random_state=123,n_estimators=n_estimators,learning_rate=0.3)
  model4.fit(X_train,y_train)
  pred_oob=model4.score(X_test,y_test)
  RMSE4.append(pred_oob)
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.plot(range(1,100),RMSE1,color='black',label='learning_rate=0.01')
plt.plot(range(1,100),RMSE2,color='green',label='learning_rate=0.05')
plt.plot(range(1,100),RMSE3,color='red',label='learning_rate=0.1')
plt.plot(range(1,100),RMSE4,color='blue',label='learning_rate=0.3')
plt.yticks([i / 100 for i in range(-20, 101, 20)])
plt.ylim(-0.2, 1)
plt.xlabel('Number of n_estimators',fontdict={'family': 'Times New Roman','size':16})
plt.ylabel(r'$R^2$',fontdict={'family': 'Times New Roman','size':16})
plt.title('GRBT',fontdict={'family': 'Times New Roman','size':16})
plt.legend()
plt.savefig('GBRT1')
Z7 = model.predict(X_train)
Z8 = model.predict(X_test)
plt.figure(figsize=(9,6))
sorted_index=model.feature_importances_.argsort()
plt.barh(range(X.shape[1]),
         model.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]),X.columns[sorted_index])
plt.xlabel('Features Importance',fontdict={'family': 'Times New Roman','size':16})
plt.ylabel('Features',fontdict={'family': 'Times New Roman','size':16})
plt.title('GradientBoostingRegressor',fontdict={'family': 'Times New Roman','size':16})
plt.tight_layout()
plt.savefig('GBRT2')
print("GBRT Training R2:", r2_score(y_train, Z7), "RMSE:", np.sqrt(mean_squared_error(y_train, Z7)), "MAE:", mean_absolute_error(y_train, Z7),"MAPE:", MAPE(y_train, Z7))
print("GBRT Testing R2:", r2_score(y_test, Z8), "RMSE:", np.sqrt(mean_squared_error(y_test, Z8)), "MAE:", mean_absolute_error(y_test, Z8),"MAPE:", MAPE(y_test, Z8))
print(model.score(X_test,y_test))
Z7 = model.predict(X_train)
Z8 = model.predict(X_test)
xx = np.linspace(-1, 120, 1000)
yy = xx
sns.set_style("whitegrid")
sns.set(style="ticks")
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.figure(figsize=(8,6))

plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(xx, yy, '--',c='grey', linewidth=1.5)
#plt.plot(xx, 1.1 * xx, '--', c='lightblue', linewidth=1.5, label='y = 1.05x')
#plt.plot(xx, 0.9 * xx, '--', c='lightcoral', linewidth=1.5, label='y = 0.95x')
plt.scatter(y_train, Z7,marker='o')
plt.scatter(y_test, Z8,c='darkorange',marker='s')

plt.tick_params (axis='both',labelsize=15)
plt.yticks(fontproperties = 'Times New Roman', size = 15)
plt.xticks(fontproperties = 'Times New Roman', size = 15)

font1 = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 20,}
plt.axis('tight')
plt.xlabel('Tested Ks(kN/mm)',fontdict={'family': 'Times New Roman','size':16})
plt.ylabel('Predicted Ks(kN/mm)',fontdict={'family': 'Times New Roman','size':16})
plt.xlim([-1,120])
plt.ylim([-1,120])
plt.title('GradientBoostingRegressor',fontdict={'family': 'Times New Roman','size':16})
plt.legend(['y = x','Training set','Testing set'], loc = 'upper left', prop={'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 14,})
plt.savefig('GBRT3')
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


shap.summary_plot(shap_values, X, plot_type="dot", show=True)
plt.ylabel(r'$\mathit{italic\_text}_\mathrm{subscript}$',fontdict={'family': 'Times New Roman','size':16})

plt.show()
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
force=shap.force_plot(explainer.expected_value, shap_values[58,:], X.iloc[58,:], feature_names=bunch.feature_names, matplotlib = True,show=False)
plt.figure(figsize=(12, 6))
a=force.set_size_inches(12, 6)
plt.show()


plt.rcParams["font.family"] = "Times New Roman"
length = len(bunch.feature_names)
n=1
for i in range(length):
    shap.dependence_plot(i, shap_values, X,feature_names=bunch.feature_names,show=False,dot_size=26 )
    plt.savefig(f'plot_{i + 1}.jpg')
    plt.close()
    n=n+1

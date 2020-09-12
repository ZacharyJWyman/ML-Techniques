# ML-Techniques
Reference Guide 

## Table of Contents

### Label Encoding

#### One Hot
usage on a dataframe: ```pd.get_dummies(df)```
  
will return a sparse array: 
```
from sklearn.preprocessing import OneHotEncoder
y = OneHotEncoder().fit_transform(x)
```
call ```.toarray()``` to convert to sparse to array 

#### Data Scaling
```from sklearn.preprocessing import StandardScaler```

### Model Selection & Implementation
```from sklearn.model_Selection import train_test_split```

#### Regression
```from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet```
```from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor```
```from xgboost import XGBRegressor```
```from lightgbm import LGMBRegressor```  
  
usage: 
```   
lin_reg = LinearRegression()
log_reg = LogisticRegression()
gb = GradientBoostingRegressor()
xgb = XGBRegressor()
rfr = RandomForestRegressor()
lgm = LGMBRegressor()
ridge = Ridge()
```
  
fitting:
```
lin_reg.fit(X_train, y_train)
```  
if error occurs consider calling ```.values``` after df
    
predicting:  
```
y_pred = lin_reg.predict(X_test)
```
   
##### Scoring
```
def print_scores(y_test, y_pred):
    print('MAE: ' + str(metrics.mean_absolute_error(y_test, y_pred)))
    print('MSE: ' + str(metrics.mean_squared_error(y_test, y_pred)))
    print('RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
    print('R-Squared: ' + str(metrics.r2_score(y_test, y_pred)))
 ```

#### Classification
```from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,  VotingClassifier```
```from sklearn.tree import DecisionTreeClassifier```
   
usage:
```
rfc = RandomForestClassifier()
ada = AdaBoostClassifier()
gbc = GradientBoostingClassifier()
vc = VotingClassifier()
dt = DecisionTreeClassifier()
```
  
fitting:
```
rfc.fit(X_train, y_train)
```  
if error occurs consider calling ```.values``` after df  
  
##### Scoring
```from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score```  
usage: ```accuracy_score(y_test, y_pred)```

#### Hyptertuning Parameters
```from sklearn.model_selection import GridSearchCV, RandomizedSearchCV```  
  
usage:
```
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': ['6' ,'8' ,'9'],
}
  
grid_search = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 5, random_state = 42)
grid_search.fit(X_train, y_train)
grid_search.best_estimator_
grid_search.best_params_
```



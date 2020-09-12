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
xgb = XGBRegresspr()
rfr = RandomForestRegressor()
lgm = LGMBRegressor()
ridge = Ridge()
```
fitting:
```
lin_reg.fit(X_train, y_train)
log_reg.fit(X_train, y_train)
gb.fit(X_train, y_train)
xgb.fit(X_train, y_train)
rfr.fit(X_train, y_train)
lgm.fit(X_train, y_train)
ridge.fit(X_train, y_train)
```
if error occurs consider calling ```.values``` after df

##### Scoring
```from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score```
usage: ```accuracy_score(y_test, y_pred)```

#### Classification
```from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,  VotingClassifier```
```from sklearn.tree import DecisionTreeClassifier```
usage: ```

##### Scoring
```from sklearn

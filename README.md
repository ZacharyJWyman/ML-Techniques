# ML-Techniques
Reference Guide 

## Table of Contents
* [Label Encoding](#Label-Encoding)  
    * [One Hot](#One-Hot)  
* [Data Scaling](#Data-Scaling)  
* [Model Selection](#Model-Selection)  
     * [Regression](#Regression)        
     * [Classification](#Classification)     
* [Hypertuning](#Hypertuning-Parameters)   
* [Stacking & Blending](#Stacking-and-Blending)  
* [Dimensionality Reduction](#Dimensionality-Reduction)


### Label Encoding

#### One Hot
usage on a dataframe: ```pd.get_dummies(df)```
  
will return a sparse array: 
```
from sklearn.preprocessing import OneHotEncoder
y = OneHotEncoder().fit_transform(x)
```
call ```.toarray()``` to convert to sparse to array 

### Data Scaling
```from sklearn.preprocessing import StandardScaler```  
usage:  
```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
  
### Model Selection & Implementation
```from sklearn.model_Selection import train_test_split```

#### Regression
```from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet```
```from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor```
```from xgboost import XGBRegressor```
```from lightgbm import LGMBRegressor```  
  
usage: 
```   
lin_reg = LinearRegression()
log_reg = LogisticRegression()
gb = GradientBoostingRegressor()
dt = DecisionTreeRegressor()
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
   
##### Regression Scoring
```from sklearn.model_selection import cross_val_score```
  
```
def print_scores(y_test, y_pred):
    print('MAE: ' + str(metrics.mean_absolute_error(y_test, y_pred)))
    print('MSE: ' + str(metrics.mean_squared_error(y_test, y_pred)))
    print('RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
    print('R-Squared: ' + str(metrics.r2_score(y_test, y_pred)))
 ```
   
```
scores = cross_val_score(lin_reg, X_train, y_train, scoring = 'neg_mean_Squared_error', cv = #)
rmse_scores = np.sqrt(-scores)
def display_scores(rmse_scores):
  print('Scores: ', rmse_scores)
  print('Mean: ', rmse_scores.mean())
  print('STD: ', rmse_scores.std())
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
  
##### Classification Scoring
```from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score```  
usage: ```accuracy_score(y_test, y_pred)```

### Hypertuning Parameters
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
  
### Stacking and Blending  
Stacking and Blending ensemble-techniques generally increase model accuracy while creating a more robust model (less sensitive to outliers).   
```from mlxtend.regressor import StackingCVRegressor```
```from mlxtend.classifier import StackingCVClassifier```  
  
Stacking:
```
stack_gen = StackingCVRegressor(regressors = (xgb, gb, lin_reg, ridge, rfr),
                               meta_regressor = rfr,
                               use_features_in_secondary = True)
stack_gen.fit(X_train.values, y_train)
y_pred = stack_gen.predict(X_test.values)
```
  
Blending:  
```
def blend(X_test):
    return ((0.25 * xgb.predict(X_test)) + \
            0.25 * rfr.predict(X_test) + \
           0.5 * stack_gen.predict(np.array(X_test)))
```
  
### Dimensionality Reduction  
PCA: 
``` 
from sklearn.decomposition import PCA
pca = PCA(n_components = #)
pca_result = pca.fit_transform(df[cols].values)
```
  
Factor Analysis:
```
from sklearn.decomposition import FactorAnalysis
same implementation as PCA
```


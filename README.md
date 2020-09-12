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

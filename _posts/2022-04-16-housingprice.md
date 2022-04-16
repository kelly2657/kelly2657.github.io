# 캘리포니아 주택가격 예측_2

원래의 전처리 과정에서 두가지 변환을 처리할 것이다.

원래의 전처리 과정은 캘리포니아 주택가격 예측_1에서 살펴볼 수 있다.

사용할 변환은 다음과 같다.

__변환 1__

중간 소득과 중간 주택 가격 사이의 상관관계 그래프에서 확인할 수 있는 수평선에 위치한 데이터를 삭제한다.

__변환 2__

강의노트에서 소개된 전처리를 통해 최종적으로 생성된 24개의 특성 중에서
중간 주택 가격과의 상관계수의 절댓값이 0.2 보다 작은 특성을 삭제한다.


```python
import sys
assert sys.version_info >= (3, 7)
```




    '0.24.2'




```python
import sys
import sklearn

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
```

## 데이터 불러오기
캘리포니아 주택 관련 데이터를 불러온다.


```python
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
```


```python
#테스트 셋 만들기
import numpy as np

np.random.seed(42)
```


```python
from sklearn.model_selection import train_test_split

# 데이터를 훈련셋과 테스트 셋으로 나눈다.
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```


```python
# 소득구간별로 나누는 속성을 나타낸다
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```


```python
# 층화표집
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
```


```python
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set),
}).sort_index()

compare_props.index.name = "Income Category"
compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                   compare_props["Overall %"] - 1)
compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                  compare_props["Overall %"] - 1)

(compare_props * 100).round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall %</th>
      <th>Stratified %</th>
      <th>Random %</th>
      <th>Strat. Error %</th>
      <th>Rand. Error %</th>
    </tr>
    <tr>
      <th>Income Category</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.98</td>
      <td>3.97</td>
      <td>4.24</td>
      <td>-0.24</td>
      <td>6.45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31.88</td>
      <td>31.88</td>
      <td>30.74</td>
      <td>-0.02</td>
      <td>-3.59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.06</td>
      <td>35.05</td>
      <td>34.52</td>
      <td>-0.01</td>
      <td>-1.53</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.63</td>
      <td>17.64</td>
      <td>18.41</td>
      <td>0.03</td>
      <td>4.42</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11.44</td>
      <td>11.46</td>
      <td>12.09</td>
      <td>0.13</td>
      <td>5.63</td>
    </tr>
  </tbody>
</table>
</div>



**데이터 되돌리기**

훈련셋과 테스트셋을 구분하기 위해 사용된 `income_cat`특성 삭제


```python
# 훈련/테스트 셋 구분을 위해 사용된 income_cat 특성 삭제
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

    C:\Users\pc\anaconda3\lib\site-packages\pandas\core\frame.py:4906: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      return super().drop(
    


```python
housing = strat_train_set.copy()
```

## 데이터 시각화


```python
# 수치형 특성간 표준 상관계수 계산 후 corr_matrix에 저장
corr_matrix = housing.corr()
```


```python
# 중간주택가격 중심으로 다른 특성 간 상관관계확인
# 중간소득과 가장 상관관계 높음
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value    1.000000
    median_income         0.688390
    total_rooms           0.137549
    housing_median_age    0.102016
    households            0.071490
    total_bedrooms        0.054707
    population           -0.020134
    longitude            -0.050813
    latitude             -0.139603
    Name: median_house_value, dtype: float64




```python
import matplotlib.pyplot as plt

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
plt.show()
```


    
![png](output_17_0.png)
    


위 산점도를 보면 어딘가 인위적으로 보인다.

여기저기서 수평선이 나타나서 그렇다.

500,000에서는 제한을 걸어놔서 그럴 것이고 다른 곳에서는 사람의 개입이 나타났을거라 예상할 수 있다.

`value_counts()`를 활용하여 어디에 데이터가 가장 많은지 확인함으로 어느 지점에서 수평선이 가장 진하게 나타나는지 살펴볼 수있다.


```python
# 아래를 확인해보면 수평선이 500001.0에서 가장 진하게 나타남을 알 수 있음
housing["median_house_value"].value_counts()
```




    500001.0    764
    137500.0    101
    162500.0     91
    112500.0     82
    187500.0     76
               ... 
    298800.0      1
    328600.0      1
    412300.0      1
    197800.0      1
    17500.0       1
    Name: median_house_value, Length: 3674, dtype: int64



__변환1__


```python
#r가장 수평선이 진한 곳의 데이터는 훈련을 혼란스럽게 할 가능성이 있으므로 지운다.
idx = housing[housing['median_house_value']==500001.0].index
housing.drop(idx, inplace=True)

# 데이터 삭제 후 변화된 산점도 확인
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
plt.show()
```


    
![png](output_22_0.png)
    


## 데이터 준비
결측치를 다른 값으로 채우거나 필요없는 데이터를 지운다

데이터 준비 과정에서 파이프라인을 만들 때 여러가지를 사용하게 되는데 현재 주피터노트북의 사이킷런에서 허용되지 않는 메서드들이 많아 실행되지 않는 동작들이 많았다.

실행되지 않는 것은 주석처리를 해 두었다.


```python
# 데이터 준비
# housing = 중간주택가격 제외한 훈련에 사용되는 특성 9개, housing_labels = 중간주택가격
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```


```python
# 데이터 정제와 스케일링을 파이프라인으로 만들면 훨씬 편하다.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

housing_num = housing.select_dtypes(include=[np.number])
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), 
                             StandardScaler())
```


```python
# 결측치를 확인해보자.
housing.isnull().sum()
```




    longitude               0
    latitude                0
    housing_median_age      0
    total_rooms             0
    total_bedrooms        168
    population              0
    households              0
    median_income           0
    ocean_proximity         0
    dtype: int64




```python
# 범주형 특성값 원핫인코딩으로 새로운 특성으로 추가
from sklearn.preprocessing import OneHotEncoder

housing_cat = housing[["ocean_proximity"]]
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot
```




    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>




```python
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```




    array([[1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           ...,
           [0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0.]])




```python
# 앞서 만든 파이프라인 활용
housing_num_prepared = num_pipeline.fit_transform(housing_num)
housing_num_prepared[:2].round(2)
```




    array([[ 0.71, -0.69,  1.86, -0.33,  0.14,  0.76,  0.25, -1.15],
           [ 0.55, -0.74,  1.54,  0.45, -0.02, -0.13,  0.04,  2.99]])




```python
# def monkey_patch_get_signature_names_out():
#     """Monkey patch some classes which did not handle get_feature_names_out()
#        correctly in 1.0.0."""
#     from inspect import Signature, signature, Parameter
#     import pandas as pd
#     from sklearn.impute import SimpleImputer
#     from sklearn.pipeline import make_pipeline, Pipeline
#     from sklearn.preprocessing import FunctionTransformer, StandardScaler

#     default_get_feature_names_out = StandardScaler.get_feature_names_out

#     if not hasattr(SimpleImputer, "get_feature_names_out"):
#       print("Monkey-patching SimpleImputer.get_feature_names_out()")
#       SimpleImputer.get_feature_names_out = default_get_feature_names_out

#     if not hasattr(FunctionTransformer, "get_feature_names_out"):
#         print("Monkey-patching FunctionTransformer.get_feature_names_out()")
#         orig_init = FunctionTransformer.__init__
#         orig_sig = signature(orig_init)

#         def __init__(*args, feature_names_out=None, **kwargs):
#             orig_sig.bind(*args, **kwargs)
#             orig_init(*args, **kwargs)
#             args[0].feature_names_out = feature_names_out

#         __init__.__signature__ = Signature(
#             list(signature(orig_init).parameters.values()) + [
#                 Parameter("feature_names_out", Parameter.KEYWORD_ONLY)])

#         def get_feature_names_out(self, names=None):
#             if self.feature_names_out is None:
#                 return default_get_feature_names_out(self, names)
#             elif callable(self.feature_names_out):
#                 return self.feature_names_out(names)
#             else:
#                 return self.feature_names_out

#         FunctionTransformer.__init__ = __init__
#         FunctionTransformer.get_feature_names_out = get_feature_names_out

# monkey_patch_get_signature_names_out()
```


```python
# df_housing_num_prepared = pd.DataFrame(housing_num_prepared, 
#                                        columns=num_pipeline.get_feature_names_out(),
#                                        index=housing_num.index)
```


```python
# 특성별 파이프라인 지정
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

# 범주형 특성 파이프라인
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([("num", num_pipeline, num_attribs),
                                   ("cat", cat_pipeline, cat_attribs),
                                  ])
```


```python
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),     # np.object 대신 object 사용
)
```


```python
housing_prepared = preprocessing.fit_transform(housing)
```


```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    # 군집화
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # fit() 함수의 반환값은 언제나 self!

    # fit() 이 찾아낸 군집별 유사도를 새로운 특성으로 추가
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    # 새롭게 생성된 특성 이름
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
```


```python
# def column_ratio(X):
#     return X[:, [0]] / X[:, [1]]

# from sklearn.preprocessing import FunctionTransformer

# def ratio_pipeline(name=None):
#     return make_pipeline(SimpleImputer(strategy="median"),
#                          FunctionTransformer(column_ratio, feature_names_out=[name]),
#                          StandardScaler())

# log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
#                              FunctionTransformer(np.log),
#                              StandardScaler())
# cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
# default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
#                                      StandardScaler())
# preprocessing = ColumnTransformer([
#         ("bedrooms_ratio", ratio_pipeline("bedrooms_ratio"), ["total_bedrooms", "total_rooms"]),
#         ("rooms_per_house", ratio_pipeline("rooms_per_house"), ["total_rooms", "households"]),
#         ("people_per_house", ratio_pipeline("people_per_house"), ["population", "households"]),
#         ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
#         ("geo", cluster_simil, ["latitude", "longitude"]),
#         ("cat", cat_pipeline, make_column_selector(dtype_include=object)),  # np.object 대신 object 사용 
#         ],
#     remainder=default_num_pipeline)  # 앞서 언급되지 않는 특성에 대한 전처리 파이프라인
```


```python
housing_prepared = preprocessing.fit_transform(housing)
```


```python
housing_prepared.shape
```




    (16512, 13)



__변환2__


```python
# 변환 2
housing_pre = pd.DataFrame()
# housing_pre[preprocessing.get_feature_names_out()] = housing_prepared
housing_pre["median_house_value"] = strat_train_set["median_house_value"]
housing_pre.shape
```




    (16512, 1)




```python
corr_matrix_2 = housing_pre.corr()
```


```python
corr_matrix_2['median_house_value'].sort_values(ascending=False)

```




    median_house_value    1.0
    Name: median_house_value, dtype: float64




```python
# 추가된 모든 특성 다 상관계수의 절댓값이 0.2보다 작다.
list_name = [x for x in corr_matrix_2.index if abs(corr_matrix_2["median_house_value"][x]) < 0.2]
print(len(list_name))
```

    0
    

## 모델 선택과 훈련

### 선형회귀모델


```python
# 선형회귀모델
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
```




    Pipeline(steps=[('columntransformer',
                     ColumnTransformer(transformers=[('pipeline-1',
                                                      Pipeline(steps=[('simpleimputer',
                                                                       SimpleImputer(strategy='median')),
                                                                      ('standardscaler',
                                                                       StandardScaler())]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001D77D33C4C0>),
                                                     ('pipeline-2',
                                                      Pipeline(steps=[('simpleimputer',
                                                                       SimpleImputer(strategy='most_frequent')),
                                                                      ('onehotencoder',
                                                                       OneHotEncoder(handle_unknown='ignore'))]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001D77D33C340>)])),
                    ('linearregression', LinearRegression())])




```python
housing_predictions = lin_reg.predict(housing)
housing_predictions[:5].round(-2) 
```




    array([136300., 472600., 176700.,  88500., 221800.])




```python
housing_labels.iloc[:5].values
```




    array([162500., 500001., 110200.,  79900., 239000.])




```python
# 오차 계산
error_ratios = housing_predictions[:5].round(-2) / housing_labels.iloc[:5].values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))
```

    -16.1%, -5.5%, 60.3%, 10.8%, -7.2%
    


```python
# 아직 높지만 강의자료에 나온 자료로 훈련했을 때보다 낮아진 것을 볼 수 있다.
from sklearn.metrics import mean_squared_error

lin_rmse = mean_squared_error(housing_labels, housing_predictions,
                              squared=False)
lin_rmse
```




    68228.989111156




```python
# 선형회귀모델에 대한 교차검증
from sklearn.model_selection import cross_val_score
lin_rmses = -cross_val_score(lin_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
pd.Series(lin_rmses).describe()
```




    count       10.000000
    mean     68283.779881
    std       2228.965871
    min      65729.393052
    25%      66442.248366
    50%      67933.589415
    75%      70007.871857
    max      71511.187771
    dtype: float64



### 결정트리회귀모델


```python
# 결정트리 회귀모델
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
```




    Pipeline(steps=[('columntransformer',
                     ColumnTransformer(transformers=[('pipeline-1',
                                                      Pipeline(steps=[('simpleimputer',
                                                                       SimpleImputer(strategy='median')),
                                                                      ('standardscaler',
                                                                       StandardScaler())]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001D77D33C4C0>),
                                                     ('pipeline-2',
                                                      Pipeline(steps=[('simpleimputer',
                                                                       SimpleImputer(strategy='most_frequent')),
                                                                      ('onehotencoder',
                                                                       OneHotEncoder(handle_unknown='ignore'))]),
                                                      <sklearn.compose._column_transformer.make_column_selector object at 0x000001D77D33C340>)])),
                    ('decisiontreeregressor',
                     DecisionTreeRegressor(random_state=42))])




```python
# 심한 과적합이 발생했음을 알 수 있다.
housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions,
                              squared=False)
tree_rmse
```




    0.0




```python
# 결정트리 모델에 대한 교차검증
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", 
                              cv=10)
```


```python
tree_rmses
```




    array([69857.19787165, 69615.87733709, 71529.36317804, 70691.30317828,
           67026.49043589, 65441.48978302, 66181.11664152, 66704.95560691,
           64086.42930914, 68914.01433184])




```python
pd.Series(tree_rmses).describe()
```




    count       10.000000
    mean     68004.823767
    std       2458.322776
    min      64086.429309
    25%      66312.076383
    50%      67970.252384
    75%      69796.867738
    max      71529.363178
    dtype: float64



### 랜덤포레스트 회귀모델


```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(n_estimators=100, random_state=42))
```


```python
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10)
```


```python
pd.Series(forest_rmses).describe()
```




    count       10.000000
    mean     48750.772391
    std       1626.692344
    min      46449.926281
    25%      47283.449545
    50%      49013.875552
    75%      49806.023666
    max      51439.771152
    dtype: float64




```python
# 훈련 셋에 대한 rmse가 현저히 낮음을 보며 이 모델이 훈련셋에 과적합되었음을 알 수 있다. 
forest_reg.fit(housing, housing_labels)
housing_predictions = forest_reg.predict(housing)

forest_rmse = mean_squared_error(housing_labels, housing_predictions,
                                 squared=False)
forest_rmse
```




    18140.91963243031



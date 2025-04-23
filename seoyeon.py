import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# 1️⃣ 데이터 불러오기 및 전처리
df = pd.read_csv('./ames.csv')

# 결측치 확인
df.isnull().sum().sort_values(ascending=False).head(20)

# '없음'을 의미하는 범주형 변수 : 'None'
none_fill_cols = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence',
    'MasVnrType', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
]
df[none_fill_cols] = df[none_fill_cols].fillna('None')

# 평균으로 채우기(LotFrontage, Latitude, Longitude)
mean_fill_cols = ['LotFrontage', 'Longitude', 'Latitude']
df[mean_fill_cols] = df[mean_fill_cols].fillna(df[mean_fill_cols].mean())

# 중앙값으로 채우기(GarageYrBlt)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())

# 최빈값으로 대체(GeoRefNo)
df['GeoRefNo'] = df['GeoRefNo'].fillna(df['GeoRefNo'].mode()[0])

# 남은 결측치 확인
df.isnull().sum()[df.isnull().sum() > 0]

# 수치형 변수 : 0으로 채움
zero_fill_cols = [
    'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea'
]
df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

# Electrical (전기 시스템 종류): 최빈값으로 채움
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# Prop_Addr (주소): 컬럼 자체 삭제(분석에 불필요)
df = df.drop(columns=['Prop_Addr'])

df.isnull().sum()[df.isnull().sum() > 0]

X = df.drop(columns=['SalePrice', 'PID'])
y = df['SalePrice']

# 수치형/범주형 나누기
X_num = X.select_dtypes(include=[np.number])
X_cat = X.select_dtypes(exclude=[np.number])


# 범주형 더미코딩
X_cat_encoded = pd.get_dummies(X_cat, drop_first=False)  

# 수치형/범주형 합침침
X_processed = pd.concat([X_num, X_cat_encoded], axis=1)


###############################


# 2️⃣ LassoCV 모델링

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

alpha = np.linspace(0, 0.5, 1000)
lasso_cv = LassoCV(alphas=alpha,
                   cv=5,
                   max_iter=1000)
lasso_cv.fit(X_processed, y)
lasso_cv.alpha_     # 아래 계산한 것들 평균내서 최적의 람다값 찾은 것
lasso_cv.mse_path_
lasso_cv.coef_
importance = pd.Series(lasso_cv.coef_, index=X_processed.columns)
importance = importance[importance != 0].sort_values(key=abs, ascending=False)
importance.head(10)

import matplotlib.pyplot as plt

importance.head(10).plot(kind='barh')
plt.xlabel('Coefficient')
plt.title('Top 10 Important Variables from Lasso')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


###############################

# 3️⃣ 변수 뽑기

import seaborn as sns
sns.heatmap(X_processed.corr(), cmap='coolwarm')

corr_matrix = X_processed.corr().abs()  # 절댓값 기준
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 상관계수 0.8 이상인 변수쌍만 뽑기
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

X_reduced = X_processed.drop(columns=to_drop)

alpha = np.linspace(0, 0.5, 1000)
lasso_cv = LassoCV(alphas=alpha,
                   cv=5,
                   max_iter=1000)
lasso_cv.fit(X_reduced, y)
lasso_cv.alpha_     # 아래 계산한 것들 평균내서 최적의 람다값 찾은 것
lasso_cv.mse_path_
lasso_cv.coef_

import matplotlib.pyplot as plt

importance = pd.Series(lasso_cv.coef_, index=X_reduced.columns)
importance = importance[importance != 0].sort_values(key=abs, ascending=False)
importance.head(10)

importance.head(10).plot(kind='barh')
plt.xlabel('Coefficient')
plt.title('Top 10 Important Variables from Lasso')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



####################################################

import numpy as np
import pandas as pd

# 쓸 변수 찾기 
df = pd.read_csv('../find-my-home/ames.csv')

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
pd.reset_option('all')

selected_columns = [
    'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'LotArea',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'Fireplaces', 'GarageCars', 'CentralAir',
    'OverallQual', 'KitchenQual', 'ExterQual', 'FireplaceQu',
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'Fence',
    'Neighborhood', 'LotFrontage', 'Condition1',
    'SalePrice'
]
df = pd.read_csv('../data/ames.csv')
df = df[selected_columns]

# 1. '없음' 의미하는 범주형 변수 → 'None'
none_fill_cols = ['FireplaceQu', 'Fence']
df[none_fill_cols] = df[none_fill_cols].fillna('None')

# 2. 평균으로 채우기 → LotFrontage
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

# 수치형 변수 : 0으로 채움
zero_fill_cols = ['TotalBsmtSF','GarageCars', 'GarageArea']
df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

# 3. 남은 결측치 다시 확인
print("남은 결측치:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# X, y 분리
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# 수치형/범주형 나누기
X_num = X.select_dtypes(include=[np.number])
X_cat = X.select_dtypes(exclude=[np.number])

# 더미코딩
X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)

# 수치형 + 범주형 결합
X_processed = pd.concat([X_num, X_cat_encoded], axis=1)
X_processed.info()


# LassoCV
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

alpha = np.linspace(0, 0.5, 1000)
lasso_cv = LassoCV(alphas=alpha,
                   cv=5,
                   max_iter=1000)
lasso_cv.fit(X_processed, y)
lasso_cv.alpha_     # 아래 계산한 것들 평균내서 최적의 람다값 찾은 것
lasso_cv.mse_path_   # 1차, 2차, 3차, 4차, 5차까지 계산한 것들..


importance = pd.Series(lasso_cv.coef_, index=X_processed.columns)
importance = importance[importance != 0].sort_values(key=abs, ascending=False)
importance.head(10)

import matplotlib.pyplot as plt

importance.head(30).plot(kind='barh')
plt.xlabel('Coefficient')
plt.title('Top 10 Important Variables from Lasso')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

non_neighborhood_cols = [col for col in X_processed.columns if not col.startswith('Neighborhood')]
lasso_coef_non_neigh = importance[~importance.index.str.startswith('Neighborhood')]

import matplotlib.pyplot as plt

lasso_coef_non_neigh.head(20).plot(kind='barh')
plt.xlabel('Coefficient')
plt.title('Top 20 Non-Neighborhood Variables from Lasso')
plt.gca().invert_yaxis()  # 위에서부터 순서대로 보기
plt.tight_layout()
plt.show()
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


#####################################################################
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from collections import defaultdict

# 📌 데이터 불러오기 및 전처리
df = pd.read_csv('../data/ames.csv')

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
df = df[selected_columns]

# 결측치 처리
df[['FireplaceQu', 'Fence']] = df[['FireplaceQu', 'Fence']].fillna('None')
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df[['TotalBsmtSF', 'GarageCars', 'GarageArea']] = df[['TotalBsmtSF', 'GarageCars', 'GarageArea']].fillna(0)

# X, y 분리
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# 수치형/범주형 구분
X_num = X.select_dtypes(include=[np.number])
X_cat = X.select_dtypes(exclude=[np.number])

# 📌 더미코딩
X_cat_encoded = pd.get_dummies(X_cat, drop_first=False)
X_processed = pd.concat([X_num, X_cat_encoded], axis=1)

# 📌 원래 변수명으로 매핑할 수 있도록 dictionary 생성
dummy_to_original = {col: col.split('_')[0] if '_' in col else col for col in X_processed.columns}

# LassoCV 학습
alpha_range = np.linspace(0.0001, 0.5, 1000)
lasso = make_pipeline(StandardScaler(), LassoCV(alphas=alpha_range, cv=5, max_iter=10000))
lasso.fit(X_processed, y)

# 계수 추출
lasso_coef = lasso.named_steps['lassocv'].coef_
coef_series = pd.Series(lasso_coef, index=X_processed.columns)

# 📌 계수가 0이 아닌 변수들만 필터링
selected = coef_series[coef_series != 0]

# 📌 원래 변수 기준으로 그룹화
grouped = defaultdict(list)
for dummy_col, coef in selected.items():
    original_var = dummy_to_original[dummy_col]
    grouped[original_var].append((dummy_col, coef))

# 📌 출력
for orig, details in grouped.items():
    print(f"\n📌 {orig} ({len(details)}개):")
    for dummy, val in details:
        print(f"   - {dummy:30}: {val: .2f}")

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
df = pd.read_csv('../data/ames.csv')

selected_columns = [
    'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'LotArea',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'Fireplaces', 'GarageCars',
    'OverallQual', 'KitchenQual', 'ExterQual', 'FireplaceQu',
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'Fence',
    'Neighborhood', 'LotFrontage', 'Condition1',
    'SalePrice'
]
df = df[selected_columns]

# 결측치 처리
none_fill_cols = ['FireplaceQu', 'Fence']
df[none_fill_cols] = df[none_fill_cols].fillna('None')
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
zero_fill_cols = ['TotalBsmtSF','GarageCars', 'GarageArea']
df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

# 🟢 Target Encoding
def target_encode(df, col, target):
    means = df.groupby(col)[target].mean()
    return df[col].map(means)

categorical_cols = ['KitchenQual', 'ExterQual', 'FireplaceQu', 
                    'Fence', 'Neighborhood', 'Condition1']

for col in categorical_cols:
    df[col + '_TE'] = target_encode(df, col, 'SalePrice')

# 🟢 원본 범주형 변수 삭제 (One-Hot Encoding 아예 안 함!)
df = df.drop(columns=categorical_cols)

# ---------------------------
# 🟢 X, y 분리 (여기서 One-Hot이 아닌 Target Encoded만 사용!)
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# ---------------------------
# 🟢 LassoCV
alpha = np.linspace(0, 0.5, 1000)
lasso_cv = LassoCV(alphas=alpha, cv=5, max_iter=1000)
lasso_cv.fit(X, y)

# ---------------------------
# 🟢 중요 변수 시각화
importance = pd.Series(lasso_cv.coef_, index=X.columns)
importance = importance[importance != 0].sort_values(key=abs, ascending=False)

plt.figure(figsize=(8, 10))
importance.head(30).plot(kind='barh')
plt.xlabel('Coefficient')
plt.title('Top Important Variables from Lasso (Target Encoded)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

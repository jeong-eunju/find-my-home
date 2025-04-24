import numpy as np
import pandas as pd

df = pd.read_excel('../find-my-home/ames_df.xlsx')

# 전처리###############################################################
# 수치형 변수 qual, cond 가중치줘서 새로운 열 추가
# 가중치 주기 위해 상관계수 분석
df[['SalePrice', 'OverallQual', 'OverallCond']].corr()  ## Qual이 상관계수 높게 나와 Qual가중치를 7로 줌

# Overall 점수 계산 (OverallQual 70%, OverallCond 30%)
df['Overall'] = df['OverallQual'] * 0.7 + df['OverallCond'] * 0.3
df


# 범주형 변수 qual, cond 가중치줘서 새로운 열 추가
# 점수화 기준 (543210 스케일)
qual_map_543210 = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1,
    'None': 0,
    'nan': 0,
    '0': 0
}

# 대상 변수
qual_vars = [
    "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond",
    "HeatingQC",
    "GarageQual", "GarageCond"
]

for col in qual_vars:
    df[col] = df[col].astype(str).replace(['nan', 'NaN', '0'], 'None')  # 예외처리 강화
    df[col + "_Score"] = df[col].map(qual_map_543210)

# 결과 일부 확인
df[[col + "_Score" for col in qual_vars]]

# 퀄리티 상관관계 확인
df[["SalePrice", "OverallQual", "OverallCond"]].corr()
df[["SalePrice", "ExterQual_Score", "ExterCond_Score"]].corr()
df[["SalePrice", "GarageQual_Score", "GarageCond_Score"]].corr()
df[["SalePrice", "BsmtQual_Score", "BsmtCond_Score"]].corr()

# 범주형 데이터 가중치 열 추가
df['Exter'] = df['ExterQual_Score'] * 0.9 + df['ExterCond_Score'] * 0.1
df['Garage'] = df['GarageQual_Score'] * 0.7 + df['GarageCond_Score'] * 0.3
df["Bsmt"] = df["BsmtQual_Score"] * 0.7 + df["BsmtCond_Score"] * 0.3
df.info()

# 예산 필터링
df = df[df['SalePrice'] >= 130000]
df = df[df['SalePrice'] <= 200000]

# x, y 분리! 
X = df.drop(columns='SalePrice')
y = (df['SalePrice'])

# X -> 수치형, 범주형 분리
num_columns = X.select_dtypes(include=['number']).columns
cat_columns = X.select_dtypes(include=['object']).columns

# 범주형은 원핫, 수치형은 스케일링 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
onehot = OneHotEncoder(handle_unknown='ignore', 
                       sparse_output=False)
X_train_cat = onehot.fit_transform(X[cat_columns])

std_scaler = StandardScaler()
X_train_num = std_scaler.fit_transform(X[num_columns])

X_train_all = np.concatenate([X_train_num, X_train_cat], axis = 1)


from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

alpha = np.linspace(0, 0.5, 1000)
lasso_cv = LassoCV(alphas=alpha,
                   cv=5,
                   max_iter=1000)
lasso_cv.fit(X_train_all, y)
lasso_cv.alpha_     # 아래 계산한 것들 평균내서 최적의 람다값 찾은 것
lasso_cv.mse_path_
lasso_cv.coef_

# 1. 원핫 범주형 변수 이름 뽑기
cat_feature_names = onehot.get_feature_names_out(cat_columns)

# 2. 전체 변수 이름 (수치형 + 범주형)
feature_names = np.concatenate([num_columns, cat_feature_names])

# 3. LassoCV에서 나온 계수와 변수이름 매칭
lasso_coef = lasso_cv.coef_

# 4. DataFrame으로 정리 + 절대값 기준 정렬
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso_coef
})
coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('AbsCoefficient', ascending=False)

print(coef_df)

# 0인 값 제거
coef_df = coef_df[coef_df['Coefficient'] != 0]
coef_df = coef_df.sort_values('Coefficient', ascending=False)


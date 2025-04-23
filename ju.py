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

#
df['Exter'] = df['ExterQual_Score'] * 0.9 + df['ExterCond_Score'] * 0.1
df['Garage'] = df['GarageQual_Score'] * 0.7 + df['GarageCond_Score'] * 0.3
df["Bsmt"] = df["BsmtQual_Score"] * 0.7 + df["BsmtCond_Score"] * 0.3
df['BsmtQual_Score'].unique()
df.info()
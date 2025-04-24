import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
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

cols_to_drop = [
    'OverallQual', 'OverallCond',
    'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond',
    'GarageQual', 'GarageCond',
    'ExterQual_Score', 'ExterCond_Score',
    'BsmtQual_Score', 'BsmtCond_Score',
    'GarageQual_Score', 'GarageCond_Score'
]

df = df.drop(columns=cols_to_drop)


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


# LassoCV -> 가격에 많은 영향을 미치는 변수 찾기
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
lasso_cv_coef = lasso_cv.coef_

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
coef_df.shape


# 예시: df_sorted는 Feature, Coefficient 등이 포함된 정리된 결과 데이터프레임
coef_df['Prefix'] = coef_df['Feature'].apply(lambda x: x.split('_')[0] if '_' in x else x)

# 그룹별로 묶기 (예: 평균/합계/갯수 등 집계도 가능)
grouped = coef_df.groupby('Prefix')

# 예시: 그룹별 Feature 개수 확인
print(grouped.size())

# 예시: 그룹별 Coefficient 총합 보기   
print(grouped['AbsCoefficient'].mean().sort_values(ascending=False))


###########################################################################

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
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

cols_to_drop = [
    'OverallQual', 'OverallCond',
    'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond',
    'GarageQual', 'GarageCond',
    'ExterQual_Score', 'ExterCond_Score',
    'BsmtQual_Score', 'BsmtCond_Score',
    'GarageQual_Score', 'GarageCond_Score', 
    'Latitude', 'Longitude'
]

df = df.drop(columns=cols_to_drop)


# 예산 필터링
df = df[df['SalePrice'] >= 130000]
df = df[df['SalePrice'] <= 200000]

##################################################################

# Important features by lasso regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. 데이터 불러오기
df = pd.read_excel('../find-my-home/ames_df.xlsx')

# 2. 품질 점수 계산 (Exter, Garage, Bsmt, Overall)
qual_map_543210 = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, 'nan': 0, '0': 0}
qual_vars = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "GarageQual", "GarageCond"]

for col in qual_vars:
    df[col] = df[col].astype(str).replace(['nan', 'NaN', '0'], 'None')
    df[col + "_Score"] = df[col].map(qual_map_543210)

df['Overall'] = df['OverallQual'] * 0.7 + df['OverallCond'] * 0.3
df['Exter'] = df['ExterQual_Score'] * 0.9 + df['ExterCond_Score'] * 0.1
df['Garage'] = df['GarageQual_Score'] * 0.7 + df['GarageCond_Score'] * 0.3
df['Bsmt'] = df['BsmtQual_Score'] * 0.7 + df['BsmtCond_Score'] * 0.3

# 3. 예산 조건 필터링 (130,000 ~ 200,000)
df = df[(df['SalePrice'] >= 130000) & (df['SalePrice'] <= 200000)]

# 4. X, y 분리
X = df.drop(columns='SalePrice')
y = df['SalePrice']

# 5. 수치형 / 범주형 분리
num_columns = X.select_dtypes(include=['number']).columns
cat_columns = X.select_dtypes(include=['object']).columns

# 6. 전처리 (OneHotEncoding + StandardScaler)
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat = onehot.fit_transform(X[cat_columns])

scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_columns])

X_all = np.concatenate([X_num, X_cat], axis=1)

# 7. LassoCV 모델 훈련
alpha = np.linspace(0, 0.5, 1000)
lasso_cv = LassoCV(alphas=alpha, cv=5, max_iter=1000)
lasso_cv.fit(X_all, y)

# 8. 변수 중요도 정리
feature_names = np.concatenate([num_columns, onehot.get_feature_names_out(cat_columns)])
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso_cv.coef_
})
coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df[coef_df['Coefficient'] != 0]
coef_df = coef_df.sort_values('Coefficient', ascending=False)

# 9. 시각화: 상위 20개 변수
top_n = 20
top_coef = coef_df.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_coef['Feature'], top_coef['Coefficient'])
plt.xlabel('Coefficient')
plt.title('Top 20 Important Features by Lasso Regression')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#######################################################################
# LASSO CV 결과 시각화

import matplotlib.pyplot as plt

top_n = 20  # 상위 20개만
top_coef = coef_df.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_coef['Feature'], top_coef['Coefficient'])
plt.xlabel('Coefficient')
plt.title('Top 20 Important Features by Lasso Regression')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#######################################################################
# Prefix 컬럼이 없으면 자동 생성
if 'Prefix' not in coef_df.columns:
    coef_df['Prefix'] = coef_df['Feature'].apply(
        lambda x: x.split('_')[0] if '_' in x else x
    )

# 그룹별 평균 절대 계수 시각화
group_mean = coef_df.groupby('Prefix')['AbsCoefficient'].mean().sort_values(ascending=True)

plt.figure(figsize=(10, 6))
group_mean.plot(kind='barh')
plt.xlabel('Mean Absolute Coefficient')
plt.title('Mean Feature Influence by Category (Prefix Grouping)')
plt.tight_layout()
plt.show()

######################################################################
# NEighborhood별 가격 분포 시각화

import seaborn as sns

plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='Neighborhood', y='SalePrice')
plt.xticks(rotation=90)
plt.title('House Price Distribution by Neighborhood')
plt.tight_layout()
plt.show()
#########################################
# 시각화: 품질 점수 vs SalePrice 산점도
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_excel('../find-my-home/ames_df.xlsx')

# 점수화 기준 (543210)
qual_map_543210 = {
    'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,
    'None': 0, 'nan': 0, '0': 0
}

# 적용할 품질 관련 변수들
qual_vars = [
    "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond",
    "HeatingQC",
    "GarageQual", "GarageCond"
]

# 예외처리 및 점수 매핑
for col in qual_vars:
    df[col] = df[col].astype(str).replace(['nan', 'NaN', '0'], 'None')
    df[col + "_Score"] = df[col].map(qual_map_543210)

# 통합 품질 점수 계산
df['Overall'] = df['OverallQual'] * 0.7 + df['OverallCond'] * 0.3
df['Exter'] = df['ExterQual_Score'] * 0.9 + df['ExterCond_Score'] * 0.1
df['Garage'] = df['GarageQual_Score'] * 0.7 + df['GarageCond_Score'] * 0.3
df['Bsmt'] = df['BsmtQual_Score'] * 0.7 + df['BsmtCond_Score'] * 0.3

# 시각화: 품질 점수 vs SalePrice 산점도
score_vars = ['Overall', 'Exter', 'Garage', 'Bsmt']
for var in score_vars:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=var, y='SalePrice')
    plt.title(f'{var} vs SalePrice')
    plt.tight_layout()
    plt.show()
############################################
# heatmap 시각화: 품질 점수와 SalePrice 간의 상관관계
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_excel('../find-my-home/ames_df.xlsx')

# 수치형 Overall 점수 계산
df['Overall'] = df['OverallQual'] * 0.7 + df['OverallCond'] * 0.3

# 품질 점수 매핑 사전
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

# 점수화 대상 컬럼
qual_vars = [
    "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond",
    "GarageQual", "GarageCond"
]

# 범주형을 점수로 매핑
for col in qual_vars:
    df[col] = df[col].astype(str).replace(['nan', 'NaN', '0'], 'None')
    df[col + "_Score"] = df[col].map(qual_map_543210)

# 상관관계 분석 대상 컬럼
selected_cols = [
    "SalePrice",
    "OverallQual", "OverallCond",
    "ExterQual_Score", "ExterCond_Score",
    "GarageQual_Score", "GarageCond_Score",
    "BsmtQual_Score", "BsmtCond_Score"
]

# 상관계수 계산
corr_matrix = df[selected_cols].corr()

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap: SalePrice & Quality Scores")
plt.tight_layout()
plt.show()
# 상관관계 히트맵
corr_matrix = df[selected_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap: SalePrice & Quality Scores")
plt.tight_layout()
plt.show()

##################################################
# 지도 시각화: Ames 주택 가격(전체 가격)
import pandas as pd
import numpy as np
import plotly.express as px

# 데이터 불러오기
df = pd.read_excel("../find-my-home/ames_df.xlsx")

# 수치형 변수 기반 가중치 컬럼 추가
df['Overall'] = df['OverallQual'] * 0.7 + df['OverallCond'] * 0.3

# 범주형 점수 매핑 정의
qual_map_543210 = {
    'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, 'nan': 0, '0': 0
}

# 점수화 대상 컬럼
qual_vars = [
    "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond",
    "HeatingQC",
    "GarageQual", "GarageCond"
]

# 범주형 점수 매핑 및 가중 평균 변수 생성
for col in qual_vars:
    df[col] = df[col].astype(str).replace(['nan', 'NaN', '0'], 'None')
    df[col + "_Score"] = df[col].map(qual_map_543210)

df['Exter'] = df['ExterQual_Score'] * 0.9 + df['ExterCond_Score'] * 0.1
df['Garage'] = df['GarageQual_Score'] * 0.7 + df['GarageCond_Score'] * 0.3
df['Bsmt'] = df['BsmtQual_Score'] * 0.7 + df['BsmtCond_Score'] * 0.3

# 지도 시각화를 위한 필터링
df = df.dropna(subset=["Latitude", "Longitude", "SalePrice"])
#df = df[(df['SalePrice'] >= 130000) & (df['SalePrice'] <= 200000)]

# 가격 정규화 (0~1 범위)
df["PriceNorm"] = (df["SalePrice"] - df["SalePrice"].min()) / (df["SalePrice"].max() - df["SalePrice"].min())

# Plotly 지도 시각화
fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    size="PriceNorm",
    color="SalePrice",
    color_continuous_scale="Viridis",
    size_max=50,
    zoom=11,
    hover_name="Neighborhood",
    hover_data={
        "SalePrice": True,
        "Exter": True,
        "Garage": True,
        "Bsmt": True,
        "Overall": True,
        "Latitude": False,
        "Longitude": False
    },
    title="🏠 Ames 주택 가격 지도 (정규화된 전체 가격 기반)",
    mapbox_style="open-street-map"
)

# 중심 위치 고정 및 마진 설정
fig.update_layout(
    mapbox_center={"lat": 42.03, "lon": -93.62},
    mapbox_zoom=12,
    margin={"r": 0, "t": 40, "l": 0, "b": 0}
)

# HTML로 저장 및 시각화 출력
fig.write_html("totalames_map_price_scaled.html")
fig.show()

#################################################################
# 지도 시각화: Ames 주택 가격(예산 가격 필터링한것)

import pandas as pd
import numpy as np
import plotly.express as px

# 데이터 불러오기
df = pd.read_excel("../find-my-home/ames_df.xlsx")  # 파일 경로는 네 환경에 맞게 수정


# 수치형 변수 기반 가중치 컬럼 추가
df['Overall'] = df['OverallQual'] * 0.7 + df['OverallCond'] * 0.3

# 범주형 점수 매핑 정의
qual_map_543210 = {
    'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, 'nan': 0, '0': 0
}

# 점수화 대상 컬럼
qual_vars = [
    "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond",
    "HeatingQC",
    "GarageQual", "GarageCond"
]

# 범주형 점수 매핑 및 가중 평균 변수 생성
for col in qual_vars:
    df[col] = df[col].astype(str).replace(['nan', 'NaN', '0'], 'None')
    df[col + "_Score"] = df[col].map(qual_map_543210)

df['Exter'] = df['ExterQual_Score'] * 0.9 + df['ExterCond_Score'] * 0.1
df['Garage'] = df['GarageQual_Score'] * 0.7 + df['GarageCond_Score'] * 0.3
df['Bsmt'] = df['BsmtQual_Score'] * 0.7 + df['BsmtCond_Score'] * 0.3

# 지도 시각화를 위한 데이터 필터링
df = df.dropna(subset=["Latitude", "Longitude", "SalePrice"])
df = df[(df['SalePrice'] >= 130000) & (df['SalePrice'] <= 200000)]

# 가격 정규화
df["PriceNorm"] = (df["SalePrice"] - df["SalePrice"].min()) / (df["SalePrice"].max() - df["SalePrice"].min())

# Plotly 지도 시각화
fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    size="PriceNorm",
    color="SalePrice",
    color_continuous_scale="Viridis",
    size_max=50,
    zoom=11,
    hover_name="Neighborhood",
    hover_data={
        "SalePrice": True,
        "Exter": True,
        "Garage": True,
        "Bsmt": True,
        "Overall": True,
        "Latitude": False,
        "Longitude": False
    },
    title="🏠 Ames 주택 가격 지도 (정규화된 필터 가격 기반)",
    mapbox_style="open-street-map"
)

# 중심 고정
fig.update_layout(
    mapbox_center={"lat": 42.03, "lon": -93.62},
    mapbox_zoom=12,
    margin={"r": 0, "t": 40, "l": 0, "b": 0}
)

# HTML 저장
map_path = "filter_ames_map_price_scaled.html"
fig.write_html(map_path)
fig.show()



###############################################

import pandas as pd
import folium
from folium.plugins import MarkerCluster

# 로컬 경로에서 전처리된 데이터 불러오기
df = pd.read_excel('../find-my-home/ames_df.xlsx')

# 원본에서도 동일하게 불러오기
original_df = pd.read_excel('../find-my-home/ames_df.xlsx')

# 위치 정보 붙이기
df['Latitude'] = original_df['Latitude']
df['Longitude'] = original_df['Longitude']

# NaN 제거
df = df.dropna(subset=['Latitude', 'Longitude'])

# ✅ 이상치 제거 (Ames 지역 근처만 남기기)
df = df[(df['Latitude'] >= 41) & (df['Latitude'] <= 43)]
df = df[(df['Longitude'] >= -94.5) & (df['Longitude'] <= -93)]

# 예산 필터링
#df = df[(df['SalePrice'] >= 130000) & (df['SalePrice'] <= 200000)]

# 지도 생성
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

# 핀 추가
locations = []
for idx, row in df.iterrows():
    loc = [row['Latitude'], row['Longitude']]
    locations.append(loc)
    folium.Marker(
        location=loc,
        popup=f"Price: ${row['SalePrice']:,} \nNeighborhood: {row['Neighborhood']}"
    ).add_to(marker_cluster)

# 핀 영역으로 자동 줌 설정
m.fit_bounds(locations)

# 저장
m.save('../find-my-home/map.html')
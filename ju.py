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
    'Latitude', 'Longitude','HeatingQC'
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



# 예산 내 최고 스펙 조합을 찾는 함수
def find_best_home_within_budget(df, model, scaler, encoder, num_cols, cat_cols, budget):
    """
    예산 내에서 가장 높은 예측 집값을 가지는 조건 조합을 찾음

    Parameters:
    - df: 전처리된 원본 데이터프레임
    - model: 훈련된 LassoCV 모델
    - scaler: 수치형 표준화 도구
    - encoder: 범주형 인코더
    - num_cols: 수치형 변수 리스트
    - cat_cols: 범주형 변수 리스트
    - budget: 예산 상한 (ex: 200000)

    Returns:
    - 최고 예측 가격과 해당 조건
    """
    best_price = -np.inf
    best_condition = None

    # 예산 내 데이터만 사용
    df_budget = df[df['SalePrice'] <= budget]

    # 중복 제거된 후보 범주 조합만 추출
    unique_combinations = df_budget[cat_cols].drop_duplicates().astype(str)

    # 수치형 평균 고정
    input_num = pd.DataFrame([df_budget[num_cols].mean()], columns=num_cols)
    input_num_scaled = scaler.transform(input_num)

    for _, row in unique_combinations.iterrows():
        input_cat = pd.DataFrame([row], columns=cat_cols).astype(str)
        encoded_cat = encoder.transform(input_cat)
        X_input = np.concatenate([input_num_scaled, encoded_cat], axis=1)
        predicted = model.predict(X_input)[0]

        if predicted > best_price:
            best_price = predicted
            best_condition = row.to_dict()

    return best_price, best_condition

# 함수 실행 (예산: $200,000)
best_price, best_condition = find_best_home_within_budget(
    df, lasso_cv, std_scaler, onehot, num_columns, cat_columns, budget=200000
)
best_price, best_condition



# 진짜 내가 원하는 옵션을 고정하고 예산 안에서 가장 좋은 조합을 추천한다면? 
# 고정 조건 포함 함수 다시 정의
def find_best_with_constraints(df, model, scaler, encoder, num_cols, cat_cols, budget, fixed_conditions):
    best_price = -np.inf
    best_condition = None

    df_budget = df[df['SalePrice'] <= budget]
    for key, value in fixed_conditions.items():
        df_budget = df_budget[df_budget[key].astype(str) == str(value)]

    if df_budget.empty:
        return None, "조건을 만족하는 집이 없습니다."

    unique_combinations = df_budget[cat_cols].drop_duplicates().astype(str)
    input_num = pd.DataFrame([df_budget[num_cols].mean()], columns=num_cols)
    for k, v in fixed_conditions.items():
        if k in num_cols:
            input_num[k] = v
    input_num_scaled = scaler.transform(input_num)

    for _, row in unique_combinations.iterrows():
        input_cat = pd.DataFrame([row], columns=cat_cols).astype(str)
        for k, v in fixed_conditions.items():
            if k in cat_cols:
                input_cat[k] = v

        encoded_cat = encoder.transform(input_cat)
        X_input = np.concatenate([input_num_scaled, encoded_cat], axis=1)
        predicted = model.predict(X_input)[0]

        if predicted > best_price:
            best_price = predicted
            best_condition = row.to_dict()
            best_condition.update({k: v for k, v in fixed_conditions.items() if k not in row})

    return best_price, best_condition

# 테스트 실행
best_price_fixed, best_condition_fixed = find_best_with_constraints(
    df=df,
    model=lasso_cv,
    scaler=std_scaler,
    encoder=onehot,
    num_cols=num_columns,
    cat_cols=cat_columns,
    budget=200000,
    fixed_conditions={
        'Neighborhood': 'Greens',
        'BldgType': '1Fam'
    }
)
best_price_fixed, best_condition_fixed







#######################################################################
import plotly.graph_objects as go

group_mean = coef_df.groupby('Prefix')['AbsCoefficient'].mean().sort_values()

fig = go.Figure(go.Bar(
    x=group_mean.values,
    y=group_mean.index,
    orientation='h',
    marker=dict(color='skyblue')
))
fig.update_layout(
    title="📊 변수 Prefix 그룹별 평균 영향력",
    template='plotly_white'
)
fig.show()




import plotly.express as px

top_20 = coef_df.head(20)
fig = px.bar(top_20.sort_values('Coefficient'), 
             x='Coefficient', y='Feature', 
             orientation='h', title='🏆 집값에 영향을 주는 Top 20 변수',
             labels={'Coefficient':'계수', 'Feature':'변수'})
fig.update_layout(yaxis=dict(tickfont=dict(size=12)), height=600)
fig.show()



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


cols_to_drop = [
    'OverallQual', 'OverallCond',
    'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond',
    'GarageQual', 'GarageCond',
    'ExterQual_Score', 'ExterCond_Score',
    'BsmtQual_Score', 'BsmtCond_Score',
    'GarageQual_Score', 'GarageCond_Score', 
    'HeatingQC'
]

df = df.drop(columns=cols_to_drop)


# 함수 실행
best_price_fixed, best_condition_fixed = find_best_with_constraints(
    df=df,
    model=lasso_cv,
    scaler=std_scaler,
    encoder=onehot,
    num_cols=num_columns,
    cat_cols=cat_columns,
    budget=200000,
    fixed_conditions={
        'Neighborhood': 'Greens',
        'BldgType': '1Fam'
    }
)
import pandas as pd
import plotly.express as px

if isinstance(best_condition_fixed, dict):

    # 예산 내 필터
    filtered_df = df[df['SalePrice'] <= 200000].copy()

    # 하이라이트 집만 추출
    highlight_df = filtered_df.copy()
    for k, v in best_condition_fixed.items():
        highlight_df = highlight_df[highlight_df[k].astype(str) == str(v)]

    filtered_df['highlight'] = '일반 집'
    highlight_df['highlight'] = '✅ 최적 조합'
    viz_df = pd.concat([filtered_df, highlight_df], ignore_index=True)

    # ====== ✅ Ames 시 중심 좌표로 강제 설정 ======
    center_lat = 42.0308
    center_lon = -93.6319

    fig = px.scatter_mapbox(
        viz_df,
        lat="Latitude",
        lon="Longitude",
        color="highlight",
        hover_name="Neighborhood",
        hover_data=["SalePrice", "Overall", "Garage", "Bsmt"],
        zoom=11,
        height=600,
        color_discrete_map={"✅ 최적 조합": "crimson", "일반 집": "lightgray"},
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        title="📍 예산 내 최적 조건 조합 지도 시각화",
        margin={"r":0,"t":40,"l":0,"b":0}
    )

    fig.show()

else:
    print("❌ 조건에 해당하는 집이 없습니다.")

























# interactivity for Quarto dashboard
# requirements: pip install ipywidgets plotly pandas scikit-learn

import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display

# 1. 드롭다운 위젯 만들기 (사용자가 원하는 옵션을 선택)
neighborhood_options = sorted(df['Neighborhood'].dropna().unique())
bldgtype_options = sorted(df['BldgType'].dropna().unique())

neighborhood_dropdown = widgets.Dropdown(
    options=neighborhood_options,
    description='지역:',
    value='Gilbert'
)

bldgtype_dropdown = widgets.Dropdown(
    options=bldgtype_options,
    description='주택유형:',
    value='1Fam'
)

# 2. 조건 고정 + 지도 업데이트 함수
def update_map(neighborhood, bldgtype):
    best_price_fixed, best_condition_fixed = find_best_with_constraints(
        df=df,
        model=lasso_cv,
        scaler=std_scaler,
        encoder=onehot,
        num_cols=num_columns,
        cat_cols=cat_columns,
        budget=200000,
        fixed_conditions={
            'Neighborhood': neighborhood,
            'BldgType': bldgtype
        }
    )

    if not isinstance(best_condition_fixed, dict):
        print("❌ 조건에 맞는 집이 없습니다.")
        return

    filtered_df = df[df['SalePrice'] <= 200000].copy()
    highlight_df = filtered_df.copy()
    for k, v in best_condition_fixed.items():
        highlight_df = highlight_df[highlight_df[k].astype(str) == str(v)]

    filtered_df['highlight'] = '일반 집'
    highlight_df['highlight'] = '✅ 최적 조합'
    viz_df = pd.concat([filtered_df, highlight_df], ignore_index=True)

    fig = px.scatter_mapbox(
        viz_df,
        lat="Latitude",
        lon="Longitude",
        color="highlight",
        hover_name="Neighborhood",
        hover_data=["SalePrice", "Overall", "Garage", "Bsmt"],
        zoom=11,
        height=600,
        color_discrete_map={"✅ 최적 조합": "crimson", "일반 집": "lightgray"},
        center={"lat": 42.0308, "lon": -93.6319},
        title=f"📍 조건 ({neighborhood}, {bldgtype}) 기반 최적 조합"
    )

    fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
    fig.show()

# 3. 연결: 상호작용 이벤트 바인딩
widgets.interact(update_map, neighborhood=neighborhood_dropdown, bldgtype=bldgtype_dropdown)

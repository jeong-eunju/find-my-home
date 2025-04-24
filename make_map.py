import pandas as pd
import folium
from folium.plugins import MarkerCluster

# 로컬 경로에서 전처리된 데이터 불러오기
df = pd.read_excel('../find-my-home/ames_df.xlsx')

# 원본에서도 동일하게 불러오기
original_df = pd.read_excel('../find-my-home/ames_df.xlsx')
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# 데이터 불러오기
df = pd.read_excel('../find-my-home/ames_df.xlsx')
original_df = pd.read_excel('../find-my-home/ames_df.xlsx')

# 위치 붙이기
df['Latitude'] = original_df['Latitude']
df['Longitude'] = original_df['Longitude']
df = df.dropna(subset=['Latitude', 'Longitude'])

# Ames 지역 외 이상치 제거
df = df[(df['Latitude'] >= 41) & (df['Latitude'] <= 43)]
df = df[(df['Longitude'] >= -94.5) & (df['Longitude'] <= -93)]

# 예산 필터링
df = df[(df['SalePrice'] >= 130000) & (df['SalePrice'] <= 200000)]

# 지도 생성
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

# 🟠 기본 핀들 (클러스터)
for idx, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Price: ${row['SalePrice']:,} \nNeighborhood: {row['Neighborhood']}"
    ).add_to(marker_cluster)

# ⭐️ 특별하게 1854번 집 (아이콘 색 다르게)
special_house = df.loc[1854]
folium.Marker(
    location=[special_house['Latitude'], special_house['Longitude']],
    popup=f"🏠 Index: 1854\nPrice: ${special_house['SalePrice']:,} \nNeighborhood: {special_house['Neighborhood']}",
    icon=folium.Icon(color='red', icon='star', prefix='fa')
).add_to(m)

# 핀 영역에 맞춰줌
m.fit_bounds([[df['Latitude'].min(), df['Longitude'].min()],
              [df['Latitude'].max(), df['Longitude'].max()]])

# 저장
m.save('../find-my-home/map_filtered_with_special.html')

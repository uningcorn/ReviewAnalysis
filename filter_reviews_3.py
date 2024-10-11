import streamlit as st
import pandas as pd
import os
from collections import Counter
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.graph_objects as go


matplotlib.rcParams['axes.unicode_minus']=False
plt.rc('font', family='Malgun Gothic')

# border-radius: 10px;
st.markdown("""
    <style>
    .custom-title {
        background-color: black;
        color: white;
        font-size: 40px;
        padding: 10px;
        text-align: center;

    }
    </style>
    <div class="custom-title">MUSINSA</div>
    """, unsafe_allow_html=True)


# Okt 인스턴스 생성
okt = Okt()

# 1. 사이즈 분석 함수 정의
size_keywords = [
    "사이즈", "정사이즈", "치수", "크다", "작다", "적당하다", "맞다", "끼다", "타이트", "여유롭다", 
    "보통", "엠", "에스", "라지", "xs", "s", "m", "l", "xl", "ws", "허리", "골반", "어깨", 
    "엉덩이", "히프", "스판끼", "xs", "s", "m", "l", "스몰", "널찍하다", "편하다", "여유", 
    "오버사이즈", "오버", "오버핏", "넉넉하다", "조이다", "핏하다", "업", "다운", "미디움", "미듐", "미디", "크게", "작게"
]

similar_words = {
    '커요': ['크다', '여유롭다', '여유', '널찍하다', '넉넉하다', '오버사이즈', '오버', '박시', '오버핏'],
    '잘 맞아요': ['정사이즈', '맞다', '적당하다', '보통'],
    '작아요': ['작다', '끼다', '타이트', '조이다', '핏하다', '붙다']
}

# 리뷰의 기본형 변환 함수
def get_lemmas(text, okt):
    return [word for word, pos in okt.pos(text, norm=True, stem=True)]

# 그룹 내 중복된 단어 제거 함수
def remove_duplicate_group_words(lemmas, similar_words):
    unique_lemmas = lemmas[:]
    for group, words in similar_words.items():
        group_words_in_review = [lemma for lemma in lemmas if lemma in words]
        if len(group_words_in_review) > 1:
            for word in group_words_in_review[1:]:
                if word in unique_lemmas:
                    unique_lemmas.remove(word)
    return unique_lemmas

# 각 그룹별 비율 계산 함수
def calculate_group_ratios(lemmatized_reviews, similar_words, size_keywords, df):
    # 사이즈 관련 키워드가 포함된 리뷰 수 계산 (분모)
    size_filtered_reviews = df['common_keywords_3_plus'].apply(lambda x: any(word in x for word in size_keywords))
    size_related_review_count = size_filtered_reviews.sum()
    total_reviews = size_related_review_count if size_related_review_count > 0 else 1
    group_ratios = {}
    for group, words in similar_words.items():
        count = 0
        for review_keywords in lemmatized_reviews[size_filtered_reviews]:
            if any(word in review_keywords for word in words):
                count += 1
        group_ratios[group] = count / total_reviews
    return group_ratios

# 사이즈 분석 함수
def analyze_size(reviews_df, size_keywords, similar_words):
    reviews_df['Processed_Review'] = reviews_df['Processed_Review'].apply(
        lambda x: x.replace('정 사이즈', '정사이즈') if pd.notnull(x) else x)
    reviews_df['Processed_Review'] = reviews_df['Processed_Review'].apply(
        lambda x: x.replace('온 버핏', '오버핏') if pd.notnull(x) else x)
    lemmatized_reviews = reviews_df['Review'].apply(lambda x: get_lemmas(x, okt))
    lemmatized_reviews_unique = lemmatized_reviews.apply(lambda x: remove_duplicate_group_words(x, similar_words))
    group_ratios = calculate_group_ratios(lemmatized_reviews_unique, similar_words, size_keywords, reviews_df)
    return group_ratios

# 2. 색감 분석 함수
def process_color_data(df):
    # 색감 관련 키워드 리스트
    color_keywords = ["색감", "색상", "색", "밝다", "어둡다", "밝은", "어두운", "무채색", "크림색", "흰색", "하얀색", "화이트", "빨간색", 
                      "회색", "검은색", "보랏빛", "네이비", "브라운", "베이지", "보라색", "검정", "블랙", "그레이", "블루", "초록색", "그린", 
                      "진하다", "오묘하다"]
    
    # '색감' 분류 (정확한 단어 매칭)
    df['color'] = df['common_keywords_3_plus'].apply(
        lambda x: '색감' if isinstance(x, list) and any(kw in color_keywords for kw in x) else '-'
    )
    
    # '색감'인 행만 필터링
    color_filtered = df[df['color'] == '색감'].copy()
    
    # 'Processed_Review_Okt' 열 생성 및 형태소 분석 적용
    color_filtered['Processed_Review_Okt'] = color_filtered['Review'].apply(
        lambda x: ' '.join(okt.morphs(x, stem=False)) if isinstance(x, str) else ''
    )
    
    # 밝다/어둡다 관련 키워드 리스트
    light_keywords = ["밝다", "밝은", "크림색", "흰색", "하얀색", "화이트", "빨간색", "베이지"]
    dark_keywords = ["어둡다", "어두운", "회색", "검은색", "보랏빛", "네이비", "브라운", "보라색", "검정", "블랙", "그레이", "블루", "초록색", "그린"]
    
    # brightness 열 추가하여 밝다/어둡다/만족 분류
    color_filtered['brightness'] = color_filtered['Processed_Review_Okt'].apply(
        lambda x: '생각보다 밝아요' if any(kw in x for kw in light_keywords)
        else ('생각보다 어두워요' if any(kw in x for kw in dark_keywords) else '만족해요')
    )
    
    # '생각보다 밝아요', '생각보다 어두워요', '만족해요'의 개수 계산
    color_counts = color_filtered['brightness'].value_counts()
    
    # 전체 행 수
    total_counts = len(color_filtered)
    
    # '생각보다 밝아요', '생각보다 어두워요', '만족해요'의 비율 계산
    color_ratios = (color_counts / total_counts) * 100
    color_ratios_formatted = color_ratios.apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
    
    # 인덱스 순서를 '생각보다 밝아요', '만족해요', '생각보다 어두워요'로 맞춤
    sorted_order = ['생각보다 밝아요', '만족해요', '생각보다 어두워요']
    color_ratios_sorted = color_ratios_formatted.reindex(sorted_order)
    
    # 비율이 없는 항목은 제외
    color_ratios_sorted = color_ratios_sorted.dropna()
    
    return color_ratios_sorted.to_dict(), total_counts

# 3. 배송 분석 함수
delivery_keywords = ["배송", "느리다", "예약배송", "예약", "출고"]
    
def is_delivery(keywords):
    # keywords is a list
    # 조건 1: delivery_keywords에 포함
    if any(k in delivery_keywords for k in keywords):
        return '배송'
    # 조건 2: '깔끔하다'와 '배송' 또는 '오다'
    if '깔끔하다' in keywords and ('배송' in keywords or '오다' in keywords):
        return '배송'
    # 조건 3: '빨리'와 '오다'
    if '빨리' in keywords and '오다' in keywords:
        return '배송'
    # 조건 4: '하루', '이틀', '일주일'과 '오다', '온', '만에'
    if any(k in ['하루', '이틀', '일주일'] for k in keywords) and any(k in ['오다', '온', '만에'] for k in keywords):
        return '배송'
    # 조건 5: '빠르다'와 '오다' 또는 '배송'
    if '빠르다' in keywords and ('오다' in keywords or '배송' in keywords):
        return '배송'
    # 조건 6: '기다리다'와 '오다' 또는 '배송'
    if '기다리다' in keywords and ('오다' in keywords or '배송' in keywords):
        return '배송'
    return '-'

def analyze_delivery(reviews_df):
    # Classify each review
    reviews_df['delivery'] = reviews_df['common_keywords_3_plus'].apply(
        lambda x: is_delivery(x) if isinstance(x, list) else '-'
    )
    # Filter only '배송' reviews
    delivery_filtered = reviews_df[reviews_df['delivery'] == '배송'].copy()
    if len(delivery_filtered) == 0:
        return {}
    # Define fast and slow keywords
    fast_keywords = ["빠르다", "빨리", "오다", "하루", "로켓", "반나절", "일찍"]
    slow_keywords = ["느리다", "예약배송", "예약", "출고", "걸리다", "늦어지다", "아쉽다", "늦다", "지연", "기다리다"]
    # Classify speed
    def classify_speed(review):
        if not isinstance(review, str):
            return '만족해요'
        if any(kw in review for kw in fast_keywords):
            return '빨라요'
        elif any(kw in review for kw in slow_keywords):
            return '느려요'
        else:
            return '만족해요'
    delivery_filtered['speed'] = delivery_filtered['Review'].apply(classify_speed)
    # Calculate ratios
    speed_counts = delivery_filtered['speed'].value_counts()
    total_deliveries = len(delivery_filtered)
    speed_ratios = (speed_counts / total_deliveries) * 100
    speed_ratios_formatted = speed_ratios.apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
    return speed_ratios_formatted.to_dict()

# 4. 두께감 분석 함수
thickness_keywords = ['두께', '원단', '원단감', '얇다', '두껍다', '두툼', '비치다', '두께감']

def classify_thickness(review):
    if not isinstance(review, str):
        return '적당해요'
    tokens = okt.morphs(review, stem=True)
    if '적당하다' in tokens:
        return '적당해요'
    elif '얇다' in tokens or '비치다' in tokens:
        return '얇아요'
    elif '두껍다' in tokens or '두툼' in tokens:
        return '두꺼워요'
    elif any(word in tokens for word in ['원단', '원단감']):
        return '적당해요'
    else:
        return '적당해요'

def analyze_thickness(reviews_df):
    # Filter reviews with thickness keywords
    thickness_filtered = reviews_df[reviews_df['common_keywords_3_plus'].apply(lambda x: any(kw in x for kw in thickness_keywords))].copy()
    if len(thickness_filtered) == 0:
        return {}
    # Classify thickness
    thickness_filtered['Thickness_Class'] = thickness_filtered['Review'].apply(classify_thickness)
    # Calculate ratios
    thickness_counts = thickness_filtered['Thickness_Class'].value_counts()
    thickness_ratios = (thickness_counts / len(thickness_filtered)) * 100
    thickness_ratios_formatted = thickness_ratios.apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
    return thickness_ratios_formatted.to_dict()

# 5. 길이감 분석 함수
def process_length_data(df):
    # 필터링할 키워드 목록 정의
    filter_keywords = ['기장', '총기장', '기장감', '길이감', '길이', '길다', '짧다']
    
    # 길이감 관련 키워드 사전 정의
    short_keywords = ['짧다', '짧아요', '미니', '애매하다', '짧은']
    moderate_keywords = ['적당하다', '완벽하다', '충분하다', '미디', '중간', '적당', '맞다']
    long_keywords = ['길다', '길어요', '롱']
    
    # 'common_keywords_3_plus'에서 필터링할 단어 포함 여부 확인 함수
    def filter_by_keywords(keywords):
        if not isinstance(keywords, list):
            return False
        return any(word in keywords for word in filter_keywords)
    
    # 길이감 관련 키워드 분류 함수
    def classify_length_from_keywords(keywords):
        if not isinstance(keywords, list):
            return '기타'
        if any(word in keywords for word in short_keywords):
            return '짧아요'
        elif any(word in keywords for word in moderate_keywords):
            return '적당해요'
        elif any(word in keywords for word in long_keywords):
            return '길어요'
        else:
            return '기타'
    
    # 'common_keywords_3_plus' 열에서 필터링할 키워드 포함된 행 추출
    df_filtered = df[df['common_keywords_3_plus'].apply(filter_by_keywords)].copy()
    
    # 필터링된 데이터에 길이감 분류 적용
    df_filtered['길이감'] = df_filtered['common_keywords_3_plus'].apply(classify_length_from_keywords)
    
    # '기타' 항목 제거
    df_filtered_no_etc = df_filtered[df_filtered['길이감'] != '기타'].copy()
    
    # 길이 카테고리별 비율 계산 (퍼센티지)
    length_distribution_no_etc = df_filtered_no_etc['길이감'].value_counts(normalize=True) * 100
    length_distribution_no_etc = length_distribution_no_etc.reindex(['짧아요', '적당해요', '길어요']).dropna()
    
    # 비율 포맷팅
    length_ratios_formatted = length_distribution_no_etc.apply(lambda x: f"{x:.2f}%")
    
    return length_ratios_formatted.to_dict(), len(df_filtered_no_etc)

# 카테고리 목록 정의
categories = ['상의', '아우터', '바지', '원피스']

# Streamlit 인터페이스 구성
st.title("리뷰 요약 필터링")

# 카테고리 선택
category = st.selectbox("카테고리 선택", categories)

# 선택한 카테고리에 따라 결과 파일 로드
# 파일 이름 패턴: 'reviews_category/{category}_결과.csv'
category_result_file = os.path.join('reviews_category', f'{category}_결과.csv')

# 파일 존재 여부 확인
if os.path.exists(category_result_file):
    category_df = pd.read_csv(category_result_file)
    # NaN 값을 빈 문자열로 대체하고, 소문자로 변환
    category_df['common_keywords_3_plus'] = category_df['common_keywords_3_plus'].fillna('').str.lower().str.strip()
    # 'common_keywords_3_plus'를 리스트로 변환
    category_df['common_keywords_3_plus'] = category_df['common_keywords_3_plus'].apply(
        lambda x: [item.strip() for item in x.split(',')] if isinstance(x, str) else []
    )
else:
    st.error(f"파일을 찾을 수 없습니다: {category_result_file}")
    st.stop()

# 상품명 선택
product_names = category_df['title'].unique()
product = st.selectbox("상품명 선택", product_names)

# 선택한 상품에 해당하는 키워드 가져오기
selected_product_keywords = category_df[category_df['title'] == product]['common_keywords_3_plus'].values
keywords = [kw.strip().lower() for kw in selected_product_keywords[0]] if len(selected_product_keywords) > 0 else []

# 해당 상품의 리뷰 파일 로드
# 파일 이름 패턴: 'keyword_merged_last/{product}_keywords.csv'
# 파일명이 title과 정확히 일치해야 합니다. 공백이나 특수문자에 주의하세요.
product_review_file = os.path.join('keyword_merged_last', f'{product}_keywords.csv')

if os.path.exists(product_review_file):
    reviews_df = pd.read_csv(product_review_file)
    # NaN 값을 빈 문자열로 대체하고, 소문자로 변환
    reviews_df['common_keywords_3_plus'] = reviews_df['common_keywords_3_plus'].fillna('').str.lower().str.strip()
    reviews_df['common_keywords_3_plus'] = reviews_df['common_keywords_3_plus'].apply(
        lambda x: ', '.join(['오버핏' if keyword.strip() == '버핏' else keyword for keyword in str(x).split(',')]) if pd.notnull(x) else x)
    # 'common_keywords_3_plus'를 리스트로 변환
    reviews_df['common_keywords_3_plus'] = reviews_df['common_keywords_3_plus'].apply(
        lambda x: [item.strip() for item in x.split(',')] if isinstance(x, str) else []
    )
    reviews_df['Review'] = reviews_df['Review'].fillna('')
else:
    st.error(f"파일을 찾을 수 없습니다: {product_review_file}")
    st.stop()

# ---------------------------
# 총평 및 추가 분석
# ---------------------------
if not reviews_df.empty:
    st.subheader(f"{product}의 총평")
    
    # 키워드를 해시태그로 변환하여 출력
    hashtag_keywords = ' '.join([f"<span style='background-color: #000000; color: #FFFFFF; padding: 4px 8px; border-radius: 12px;'>#{keyword}</span>" for keyword in keywords if keyword])
    st.markdown(f"**키워드:** {hashtag_keywords}", unsafe_allow_html=True)
    
    # ---------------------------
    # 추가 분석 시작
    # ---------------------------
    
    # 1. 사이즈 (Size) 분석
    size_analysis = analyze_size(reviews_df, size_keywords, similar_words)
    
    # 2. 색감 (Color) 분석
    color_analysis, total_color_reviews = process_color_data(reviews_df)
    
    # 3. 배송 (Delivery) 분석   
    delivery_analysis = analyze_delivery(reviews_df)
    
    # 4. 두께감 (Thickness) 분석 (아우터, 상의에 한함)    
    thickness_analysis = None
    if category in ['아우터', '상의']:
        thickness_analysis = analyze_thickness(reviews_df)
    
    # 5. 길이감 (Length) 분석 (바지, 원피스에 한함)
    length_analysis = None
    if category in ['바지', '원피스']:
        length_analysis, total_length_reviews = process_length_data(reviews_df)

    # ---------------------------
    # 추가 분석 끝
    # ---------------------------
    
    # ---------------------------
    # 분석 결과 출력
    # ---------------------------
    st.markdown("---")  # Separator
    
    # 4개의 열 생성
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 색감")
        if color_analysis and total_color_reviews > 0:
            # Plotly 가로 막대그래프
            fig_color = go.Figure(go.Bar(
                x=list(color_analysis.values()), 
                y=list(color_analysis.keys()), 
                orientation='h',
                marker=dict(color='black')
            ))
            fig_color.update_layout(
                xaxis_title='비율 (%)',
                title='색감 밝기 비율',
                height=300,  # 높이 설정
                width=150,   # 너비 설정
                showlegend=False,
                yaxis=dict(
                    tickfont=dict(size=10)  # y축 글씨 크기 설정 (여기서 10을 원하는 크기로 변경)
                )
            )
            st.plotly_chart(fig_color)
            
            # 표 출력
            color_table = pd.DataFrame({
                '색감': list(color_analysis.keys()),
                '비율 (%)': [ratio for ratio in list(color_analysis.values())]
            })
            st.table(color_table)
        else:
            st.write("색감 관련 리뷰가 없습니다.")
    
    with col2:
        st.markdown("### 사이즈")
        if size_analysis:
             # Plotly 가로 막대그래프
            fig_size = go.Figure(go.Bar(
                x=[ratio * 100 for ratio in size_analysis.values()],
                y=list(size_analysis.keys()),
                orientation='h',
                marker=dict(color='black')
            ))
            fig_size.update_layout(
                xaxis_title='비율 (%)',
                title='사이즈 비율',
                height=300,  # 높이 설정
                width=150,   # 너비 설정
                showlegend=False,
                yaxis=dict(
                    tickfont=dict(size=10)  # y축 글씨 크기 설정 (여기서 10을 원하는 크기로 변경)
                )
            )
            st.plotly_chart(fig_size)
            
            # 표 출력
            size_table = pd.DataFrame({
                '사이즈': list(size_analysis.keys()),
                '비율 (%)': [f"{ratio * 100:.2f}%" for ratio in size_analysis.values()]
            })
            st.table(size_table)
        else:
            st.write("사이즈 관련 리뷰가 없습니다.")
    
    with col3:
        st.markdown("### 배송")
        if delivery_analysis:
            # Plotly 가로 막대그래프
            fig_delivery = go.Figure(go.Bar(
                x=[float(r.replace('%', '')) for r in list(delivery_analysis.values())],
                y=list(delivery_analysis.keys()),
                orientation='h',
                marker=dict(color='black')
            ))
            fig_delivery.update_layout(
                xaxis_title='비율 (%)',
                title='배송 속도 비율',
                height=300,  # 높이 설정
                width=150,   # 너비 설정
                showlegend=False,
                yaxis=dict(
                    tickfont=dict(size=10)  # y축 글씨 크기 설정 (여기서 10을 원하는 크기로 변경)
                )
            )
            st.plotly_chart(fig_delivery)
            
            # 표 출력
            delivery_table = pd.DataFrame({
                '배송 속도': list(delivery_analysis.keys()),
                '비율 (%)': [f"{float(r.replace('%', '')):.2f}%" for r in list(delivery_analysis.values())]
            })
            st.table(delivery_table)
        else:
            st.write("배송 관련 리뷰가 없습니다.")
    
    with col4:
        if category in ['아우터', '상의'] and thickness_analysis:
            st.markdown("### 두께감")
            # Plotly 가로 막대그래프
            fig_thickness = go.Figure(go.Bar(
                x=[float(r.replace('%', '')) for r in list(thickness_analysis.values())],
                #y=list(thickness_analysis.keys()),
                orientation='h',
                marker=dict(color='black')
            ))
            fig_thickness.update_layout(
                xaxis_title='비율 (%)',
                title='두께감 비율',
                height=300,  # 높이 설정
                width=150,   # 너비 설정
                showlegend=False,
                yaxis=dict(
                    tickfont=dict(size=10)  # y축 글씨 크기 설정 (여기서 10을 원하는 크기로 변경)
                )
            )
            st.plotly_chart(fig_thickness)
            
            # 표 출력
            thickness_table = pd.DataFrame({
                '두께감': list(thickness_analysis.keys()),
                '비율 (%)': [f"{float(r.replace('%', '')):.2f}%" for r in list(thickness_analysis.values())]
            })
            st.table(thickness_table)
        elif category in ['바지', '원피스'] and length_analysis:
            st.markdown("### 길이감")
            if length_analysis and total_length_reviews > 0:
                # Plotly 가로 막대그래프
                fig_length = go.Figure(go.Bar(
                    x=list(length_analysis.values()),
                    y=list(length_analysis.keys()),
                    orientation='h',
                    marker=dict(color='black')
                ))
                fig_length.update_layout(
                    xaxis_title='비율 (%)',
                    title='길이감 비율',
                    height=300,  # 높이 설정
                    width=150,
                    showlegend=False,
                    yaxis=dict(
                    tickfont=dict(size=10)  # y축 글씨 크기 설정 (여기서 10을 원하는 크기로 변경)
                    )
                )
                st.plotly_chart(fig_length)
                
                # 표 출력
                length_table = pd.DataFrame({
                    '길이감': list(length_analysis.keys()),
                    '비율 (%)': [ratio for ratio in list(length_analysis.values())]
                })
                st.table(length_table)
            else:
                st.write("길이감 관련 리뷰가 없습니다.")
    
    
    # ---------------------------
    # 분석 결과 끝
    # ---------------------------
    
else:
    st.write("해당 상품에 대한 리뷰가 없습니다.")

# ---------------------------
# 키워드 선택을 통한 리뷰 필터링
# ---------------------------
if not reviews_df.empty:
    st.subheader("키워드로 리뷰 필터링하기")
    
    # 키워드 선택 (멀티 셀렉트)
    selected_keywords = st.multiselect("키워드 선택", keywords)
    
    # 리뷰 필터링 함수 정의
    def filter_reviews_by_keywords(reviews, selected_keywords):
        if not selected_keywords:
            return reviews
        # 'common_keywords_3_plus'는 리스트 형태
        filtered = reviews[reviews['common_keywords_3_plus'].apply(
            lambda kws: any(keyword in kws for keyword in selected_keywords)
        )]
        return filtered
    
    # 필터링된 리뷰 데이터프레임
    filtered_reviews_df = filter_reviews_by_keywords(reviews_df, selected_keywords)
    
    # 필터링된 리뷰 수 강조해서 출력 (배경색 추가) 'background-color: #f0f0f5; color: #FF6347
    filtered_count = len(filtered_reviews_df)
    st.markdown(f"""
        <p style="font-size:18px;">
            필터된 리뷰 수: <span style='background-color: #f0f0f5; color: #FF6347; padding: 4px 8px; border-radius: 12px;'>{filtered_count}</span>
        </p>
        """, unsafe_allow_html=True)

    # 필터된 리뷰 출력
    st.subheader(f"{product}에 대한 리뷰")

    if not filtered_reviews_df.empty:
        for idx, row in filtered_reviews_df.iterrows():
            # 키워드를 해시태그로 변환하여 표시
            review_keywords = ' '.join([f"<span style='background-color: #000000; color: #FFFFFF; padding: 4px 8px; border-radius: 12px;'>#{kw.strip()}</span>" for kw in row['common_keywords_3_plus'] if kw.strip()])
            
            # 리뷰와 해시태그 출력
            st.write(f"{row['Review']}")
            st.markdown(review_keywords, unsafe_allow_html=True)
        
            # 각 리뷰 사이에 수평선 추가
            st.markdown("---")
    else:
        st.write("선택한 키워드에 해당하는 리뷰가 없습니다.")

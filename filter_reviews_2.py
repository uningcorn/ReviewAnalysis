import streamlit as st
import pandas as pd
import os
from collections import Counter
from konlpy.tag import Okt

# Okt 인스턴스 생성
okt = Okt()

# 카테고리 목록 정의
categories = ['상의', '아우터', '바지', '원피스']

# Streamlit 인터페이스 구성
st.title("리뷰 필터링 웹 애플리케이션")

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
    # 'common_keywords_3_plus'를 리스트로 변환
    reviews_df['common_keywords_3_plus'] = reviews_df['common_keywords_3_plus'].apply(
        lambda x: [item.strip() for item in x.split(',')] if isinstance(x, str) else []
    )
    reviews_df['Review'] = reviews_df['Review'].fillna('')
else:
    st.error(f"파일을 찾을 수 없습니다: {product_review_file}")
    st.stop()

# 총평 및 키워드 출력
if not reviews_df.empty:
    st.subheader(f"{product}의 총평")
    
    # 키워드를 해시태그로 변환하여 출력
    hashtag_keywords = ' '.join([f"#{keyword}" for keyword in keywords if keyword])
    st.markdown(f"**키워드:** {hashtag_keywords}")
    
    # ---------------------------
    # 추가 분석 시작
    # ---------------------------
    
    # 1. 사이즈 (Size) 분석
    size_keywords = [
        "사이즈", "정사이즈", "치수", "크다", "작다", "적당하다", "맞다", "끼다", "타이트", "여유롭다", 
        "보통", "엠", "에스", "라지", "xs", "s", "m", "l", "xl", "ws", "허리", "골반", "어깨", 
        "엉덩이", "히프", "스판끼", "xs", "s", "m", "l", "스몰", "널찍하다", "편하다", "여유", 
        "오버사이즈", "오버", "오버핏", "넉넉하다", "조이다", "핏하다", "업", "다운", "미디움", "미듐", "미디", "크게", "작게"
    ]
    
    similar_words = {
        '정사이즈': ['정사이즈', '맞다', '적당하다', '보통'],
        '크다': ['크다', '여유롭다', '여유', '널찍하다', '넉넉하다', '오버사이즈', '오버', '박시', '오버핏'],
        '작다': ['작다', '끼다', '타이트', '조이다', '핏하다', '붙다']
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
    def calculate_group_ratios(reviews, similar_words, size_keywords, df):
        # 사이즈 관련 키워드가 포함된 리뷰 수 계산 (분모)
        size_filtered_reviews = df['common_keywords_3_plus'].apply(lambda x: any(word in x for word in size_keywords))
        size_related_review_count = size_filtered_reviews.sum()
        total_reviews = size_related_review_count if size_related_review_count > 0 else 1
        group_ratios = {}
        for group, words in similar_words.items():
            count = 0
            for review_keywords in reviews[size_filtered_reviews]:
                if any(word in review_keywords for word in words):
                    count += 1
            group_ratios[group] = count / total_reviews
        return group_ratios
    
    # 사이즈 분석 함수
    def analyze_size(reviews_df):
        lemmatized_reviews = reviews_df['Review'].apply(lambda x: get_lemmas(x, okt))
        lemmatized_reviews_unique = lemmatized_reviews.apply(lambda x: remove_duplicate_group_words(x, similar_words))
        group_ratios = calculate_group_ratios(lemmatized_reviews_unique, similar_words, size_keywords, reviews_df)
        return group_ratios
    
    size_analysis = analyze_size(reviews_df)
    
    # 2. 색감 (Color) 분석
    color_keywords = ["색감", "색상", "색", "밝다", "어둡다", "밝은", "어두운", "무채색", "크림색", "흰색", "하얀색", "화이트", "빨간색", 
                      "회색", "검은색", "보랏빛", "네이비", "브라운", "베이지", "보라색", "검정", "블랙", "그레이", "블루", "초록색", "그린", 
                      "진하다", "오묘하다"]

    def analyze_color(reviews_df):
        # Filter reviews with color keywords
        color_filtered = reviews_df[reviews_df['common_keywords_3_plus'].apply(lambda x: any(kw in x for kw in color_keywords))]
        total_color_reviews = len(color_filtered)
        if total_color_reviews == 0:
            return [], 0
        # Collect color keywords
        all_color_keywords = [kw for kws in color_filtered['common_keywords_3_plus'] for kw in kws if kw in color_keywords]
        color_counts = Counter(all_color_keywords)
        most_common_colors = color_counts.most_common(5)
        return most_common_colors, total_color_reviews

    color_analysis, total_color_reviews = analyze_color(reviews_df)

    # 3. 배송 (Delivery) 분석
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
        delivery_filtered = reviews_df[reviews_df['delivery'] == '배송']
        if len(delivery_filtered) == 0:
            return {}
        # Define fast and slow keywords
        fast_keywords = ["빠르다", "빨리", "오다", "하루", "로켓", "반나절", "일찍"]
        slow_keywords = ["느리다", "예약배송", "예약", "출고", "걸리다", "늦어지다", "아쉽다", "늦다", "지연", "기다리다"]
        # Classify speed
        def classify_speed(review):
            if not isinstance(review, str):
                return '만족하다'
            if any(kw in review for kw in fast_keywords):
                return '빠르다'
            elif any(kw in review for kw in slow_keywords):
                return '느리다'
            else:
                return '만족하다'
        delivery_filtered['speed'] = delivery_filtered['Review'].apply(classify_speed)
        # Calculate ratios
        speed_counts = delivery_filtered['speed'].value_counts()
        total_deliveries = len(delivery_filtered)
        speed_ratios = (speed_counts / total_deliveries) * 100
        speed_ratios_formatted = speed_ratios.apply(lambda x: f"{x:.2f}%")
        return speed_ratios_formatted.to_dict()

    delivery_analysis = analyze_delivery(reviews_df)

    # 4. 두께감 (Thickness) 분석 (아우터, 상의에 한함)
    thickness_keywords = ['두께', '원단', '원단감', '얇다', '두껍다', '두툼', '비치다', '두께감']
    similar_thickness_words = {
        '적당하다': ['적당하다'],
        '얇다': ['얇다', '비치다'],
        '두껍다': ['두껍다', '두툼']
    }

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
        thickness_filtered = reviews_df[reviews_df['common_keywords_3_plus'].apply(lambda x: any(kw in x for kw in thickness_keywords))]
        if len(thickness_filtered) == 0:
            return {}
        # Classify thickness
        thickness_filtered['Thickness_Class'] = thickness_filtered['Review'].apply(classify_thickness)
        # Calculate ratios
        thickness_counts = thickness_filtered['Thickness_Class'].value_counts()
        thickness_ratios = (thickness_counts / len(thickness_filtered)) * 100
        thickness_ratios_formatted = thickness_ratios.apply(lambda x: f"{x:.2f}%")
        return thickness_ratios_formatted.to_dict()

    thickness_analysis = None
    if category in ['아우터', '상의']:
        thickness_analysis = analyze_thickness(reviews_df)

    # 5. 길이감 (Length) 분석 (바지, 원피스에 한함)
    length_keywords = ['기장', '총기장', '기장감', '길이감', '길이', '길다', '짧다']
    similar_length_words = {
        '짧아요': ['짧다', '짧아요', '미니', '애매하다', '짧은'],
        '적당해요': ['적당하다', '완벽하다', '충분하다', '미디', '중간', '적당', '맞다'],
        '길어요': ['길다', '길어요', '롱']
    }

    def classify_length(keywords):
        if not isinstance(keywords, list):
            return '기타'
        if any(word in keywords for word in ['짧다', '짧아요', '미니', '애매하다', '짧은']):
            return '짧아요'
        elif any(word in keywords for word in ['적당하다', '완벽하다', '충분하다', '미디', '중간', '적당', '맞다']):
            return '적당해요'
        elif any(word in keywords for word in ['길다', '길어요', '롱']):
            return '길어요'
        else:
            return '기타'

    def analyze_length(reviews_df):
        # Filter reviews with length keywords
        length_filtered = reviews_df[reviews_df['common_keywords_3_plus'].apply(lambda x: any(kw in x for kw in length_keywords))]
        if len(length_filtered) == 0:
            return {}
        # Classify length
        length_filtered['Length_Class'] = length_filtered['common_keywords_3_plus'].apply(classify_length)
        # Remove '기타'
        length_filtered = length_filtered[length_filtered['Length_Class'] != '기타']
        if len(length_filtered) == 0:
            return {}
        # Calculate ratios
        length_counts = length_filtered['Length_Class'].value_counts()
        length_ratios = (length_counts / len(length_filtered)) * 100
        length_ratios_formatted = length_ratios.apply(lambda x: f"{x:.2f}%")
        return length_ratios_formatted.to_dict()

    length_analysis = None
    if category in ['바지', '원피스']:
        length_analysis = analyze_length(reviews_df)

    # 6. 색감 (Color) 분석 (추가적으로 밝기 분류)
    def analyze_color_with_classification(reviews_df):
        color_filtered = reviews_df[reviews_df['common_keywords_3_plus'].apply(lambda x: any(kw in x for kw in color_keywords))]
        if len(color_filtered) == 0:
            return {}, 0
        # Use Okt to morph words
        color_filtered['Processed_Review_Okt'] = color_filtered['Review'].apply(lambda x: ' '.join(okt.morphs(x, stem=False)) if isinstance(x, str) else '')
        # Define light and dark keywords
        light_keywords = ["밝다", "밝은", "크림색", "흰색", "하얀색", "화이트", "빨간색", "베이지"]
        dark_keywords = ["어둡다", "어두운", "회색", "검은색", "보랏빛", "네이비", "브라운", "보라색", "검정", "블랙", "그레이", "블루", "초록색", "그린"]
        # Classify brightness
        def classify_brightness(review):
            if not isinstance(review, str):
                return '만족해요'
            if any(kw in review for kw in light_keywords):
                return '생각보다 밝아요'
            elif any(kw in review for kw in dark_keywords):
                return '생각보다 어두워요'
            else:
                return '만족해요'
        color_filtered['brightness'] = color_filtered['Processed_Review_Okt'].apply(classify_brightness)
        # Count
        brightness_counts = color_filtered['brightness'].value_counts()
        total = len(color_filtered)
        brightness_ratios = (brightness_counts / total) * 100
        brightness_ratios_formatted = brightness_ratios.apply(lambda x: f"{x:.2f}%")
        # Sort order
        sorted_order = ['생각보다 밝아요', '만족해요', '생각보다 어두워요']
        brightness_ratios_sorted = brightness_ratios_formatted.reindex(sorted_order)
        # Remove NaN
        brightness_ratios_sorted = brightness_ratios_sorted.dropna()
        return brightness_ratios_sorted.to_dict(), total

    # Analyze color with brightness classification
    color_brightness_ratios, total_color_reviews = analyze_color_with_classification(reviews_df)

    # ---------------------------
    # 추가 분석 끝
    # ---------------------------
    
    # ---------------------------
    # 분석 결과 출력
    # ---------------------------
    st.markdown("---")  # Separator

    # 사이즈 분석 결과
    st.markdown("### 사이즈 관련 분석")
    if size_analysis:
        for group, ratio in size_analysis.items():
            st.write(f"- **{group}**: {ratio*100:.2f}%")
    else:
        st.write("사이즈 관련 리뷰가 없습니다.")

    # 색감 분석 결과
    st.markdown("### 색감 관련 분석")
    if color_brightness_ratios and total_color_reviews > 0:
        for cls, ratio in color_brightness_ratios.items():
            st.write(f"- **{cls}**: {ratio}")
    else:
        st.write("색감 관련 리뷰가 없습니다.")

    # 배송 분석 결과
    st.markdown("### 배송 관련 분석")
    if delivery_analysis:
        for speed, ratio in delivery_analysis.items():
            st.write(f"- **{speed}**: {ratio}")
    else:
        st.write("배송 관련 리뷰가 없습니다.")

    # 두께감 분석 결과
    if thickness_analysis:
        st.markdown("### 두께감 관련 분석")
        for cls, ratio in thickness_analysis.items():
            st.write(f"- **{cls}**: {ratio}")
    
    # 길이감 분석 결과
    if length_analysis:
        st.markdown("### 길이감 관련 분석")
        for cls, ratio in length_analysis.items():
            st.write(f"- **{cls}**: {ratio}")

    # ---------------------------
    # 필터링된 리뷰 출력
    # ---------------------------
    st.markdown("---")  # Separator
    st.subheader(f"{product}에 대한 리뷰")
    if not reviews_df.empty:
        # 필터링된 리뷰 데이터프레임
        filtered_reviews_df = reviews_df.copy()
        
        if selected_keywords:
            # 키워드 필터링
            filtered_reviews_df = filtered_reviews_df[filtered_reviews_df['common_keywords_3_plus'].apply(
                lambda kws: any(keyword in kws for keyword in selected_keywords)
            )]
        
        if not filtered_reviews_df.empty:
            for idx, row in filtered_reviews_df.iterrows():
                # 키워드를 해시태그로 변환하여 표시
                review_keywords = ' '.join([f"#{kw.strip()}" for kw in row['common_keywords_3_plus'] if kw.strip()])
                st.write(f"- {row['Review']} {review_keywords}")
        else:
            st.write("선택한 키워드에 해당하는 리뷰가 없습니다.")
    else:
        st.write("해당 상품에 대한 리뷰가 없습니다.")

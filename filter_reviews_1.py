import streamlit as st
import pandas as pd
import os

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
else:
    st.error(f"파일을 찾을 수 없습니다: {category_result_file}")
    st.stop()

# 상품명 선택
product_names = category_df['title'].unique()
product = st.selectbox("상품명 선택", product_names)

# 선택한 상품에 해당하는 키워드 가져오기
selected_product_keywords = category_df[category_df['title'] == product]['common_keywords_3_plus'].values
keywords = [kw.strip().lower() for kw in selected_product_keywords[0].split(',')] if len(selected_product_keywords) > 0 else []

# 해당 상품의 리뷰 파일 로드
# 파일 이름 패턴: 'keyword_merged_last/{product}_keywords.csv'
# 파일명이 title과 정확히 일치해야 합니다. 공백이나 특수문자에 주의하세요.
product_review_file = os.path.join('keyword_merged_last', f'{product}_keywords.csv')

if os.path.exists(product_review_file):
    reviews_df = pd.read_csv(product_review_file)
    # NaN 값을 빈 문자열로 대체하고, 소문자로 변환
    reviews_df['common_keywords_3_plus'] = reviews_df['common_keywords_3_plus'].fillna('').str.lower().str.strip()
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
else:
    st.write("해당 상품에 대한 리뷰가 없습니다.")

# 키워드 선택을 통한 리뷰 필터링
st.subheader("키워드로 리뷰 필터링하기")

# 키워드 선택 (멀티 셀렉트)
selected_keywords = st.multiselect("키워드 선택", keywords)

# 리뷰 필터링 함수 정의
def filter_reviews_by_keywords(reviews, selected_keywords):
    if not selected_keywords:
        return reviews
    # 키워드가 쉼표로 구분된 문자열로 저장되어 있다고 가정
    filtered = reviews[reviews['common_keywords_3_plus'].apply(
        lambda kws: any(keyword in [k.strip() for k in kws.split(',')] for keyword in selected_keywords)
    )]
    return filtered

# 필터링된 리뷰 데이터프레임
filtered_reviews_df = filter_reviews_by_keywords(reviews_df, selected_keywords)

# 필터된 리뷰 출력
st.subheader(f"{product}에 대한 리뷰")

if not filtered_reviews_df.empty:
    for idx, row in filtered_reviews_df.iterrows():
        # 키워드를 해시태그로 변환하여 표시
        review_keywords = ' '.join([f"#{kw.strip()}" for kw in row['common_keywords_3_plus'].split(',') if kw.strip()])
        st.write(f"- {row['Review']} {review_keywords}")
else:
    st.write("선택한 키워드에 해당하는 리뷰가 없습니다.")

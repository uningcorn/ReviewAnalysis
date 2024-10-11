import streamlit as st
import pandas as pd

import plotly.express as px


# HTML과 CSS를 사용해 타이틀 스타일링
st.markdown("""
    <style>
    .custom-title {
        background-color: black;
        color: white;
        font-size: 40px;
        padding: 10px;
        text-align: center;
        border-radius: 10px;
    }
    </style>
    <div class="custom-title">MUSINSA</div>
    """, unsafe_allow_html=True)

# # CSV 파일 불러오기
# @st.cache_data
# def load_data():
#     df = pd.read_csv('merged_결과.csv')
#     return df

# # 데이터 불러오기
# df = load_data()

# # common_keywords_3_plus에 있는 모든 키워드를 수집
# def extract_all_keywords(df):
#     all_keywords = set()
#     for keywords in df['common_keywords_3_plus']:
#         keyword_list = [keyword.strip() for keyword in keywords.split(',')]
#         all_keywords.update(keyword_list)
#     return list(all_keywords)

# # 모든 키워드 추출
# all_keywords = extract_all_keywords(df)

# # 세션 상태를 사용해 검색 키워드 저장
# if 'search_keyword' not in st.session_state:
#     st.session_state['search_keyword'] = ""

# # 검색창 만들기
# st.title("Common Keywords Search")

# # 검색어 입력 (추천 키워드를 클릭하면 자동으로 세션에 저장됨)
# search_keyword = st.text_input("Search common keywords:", st.session_state['search_keyword'])

# # 추천 키워드 개수 설정
# num_recommendations = 5

# # 사용자가 추천된 키워드를 누르면 그 키워드로 검색을 수행하기 위한 함수
# def search_by_keyword(search_keyword):
#     filtered_df = df[df['common_keywords_3_plus'].apply(lambda x: search_keyword in [keyword.strip() for keyword in x.split(',')])]
#     return filtered_df

# # 추천된 키워드를 보여주고 클릭할 수 있도록 함
# def show_recommendations(recommendations):
#     st.write("Did you mean one of these?")
#     for rec in recommendations:
#         if st.button(rec):
#             # 버튼을 누르면 세션 상태에 선택된 추천 키워드를 저장하고 화면을 새로고침
#             st.session_state['search_keyword'] = rec
#             st.experimental_rerun()

# # 검색어가 입력된 경우
# if search_keyword:
#     # common_keywords_3_plus 열에서 ,로 분리된 키워드 중 검색어가 포함된 행 필터링
#     filtered_df = search_by_keyword(search_keyword)
    
#     # 결과 보여주기
#     if not filtered_df.empty:
#         st.write(f"Results for '{search_keyword}':")
#         st.dataframe(filtered_df)
#     else:
#         st.write(f"No exact matches found for '{search_keyword}'.")

#     # 추천 키워드 찾기 (결과 유무와 관계없이 항상 추천)
#     recommendations = difflib.get_close_matches(search_keyword, all_keywords, n=num_recommendations, cutoff=0.5)
    
#     if recommendations:
#         show_recommendations(recommendations)
#     else:
#         st.write("No similar keywords found.")
# else:
#     st.write("Enter a keyword to search.")

# # 전체 데이터 미리보기 (옵션)
# st.write("Preview of merged_결과.csv:")
# st.dataframe(df.head())


# 사이드바에 넣어버리기
import streamlit as st
import pandas as pd
import difflib

# CSV 파일 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv('merged_결과.csv')
    return df

# 데이터 불러오기
df = load_data()

# common_keywords_3_plus에 있는 모든 키워드를 수집
def extract_all_keywords(df):
    all_keywords = set()
    for keywords in df['common_keywords_3_plus']:
        keyword_list = [keyword.strip() for keyword in keywords.split(',')]
        all_keywords.update(keyword_list)
    return list(all_keywords)

# 모든 키워드 추출
all_keywords = extract_all_keywords(df)

# 세션 상태를 사용해 검색 키워드 저장
if 'search_keyword' not in st.session_state:
    st.session_state['search_keyword'] = ""

# 사이드바 검색창 만들기
st.sidebar.title("Common Keywords Search")

# 사이드바에서 검색어 입력 (추천 키워드를 클릭하면 자동으로 세션에 저장됨)
search_keyword = st.sidebar.text_input("Search common keywords:", st.session_state['search_keyword'])

# 추천 키워드 개수 설정
num_recommendations = 5

# 사용자가 추천된 키워드를 누르면 그 키워드로 검색을 수행하기 위한 함수
def search_by_keyword(search_keyword):
    filtered_df = df[df['common_keywords_3_plus'].apply(lambda x: search_keyword in [keyword.strip() for keyword in x.split(',')])]
    return filtered_df

# 사이드바에 추천된 키워드를 보여주고 클릭할 수 있도록 함
def show_recommendations(recommendations):
    st.sidebar.write("Did you mean one of these?")
    for rec in recommendations:
        if st.sidebar.button(rec):
            # 버튼을 누르면 세션 상태에 선택된 추천 키워드를 저장하고 화면을 새로고침
            st.session_state['search_keyword'] = rec
            st.experimental_rerun()

# 검색어가 입력된 경우
if search_keyword:
    # common_keywords_3_plus 열에서 ,로 분리된 키워드 중 검색어가 포함된 행 필터링
    filtered_df = search_by_keyword(search_keyword)
    
    # 결과 보여주기 (title만 출력)
    if not filtered_df.empty:
        st.write(f"Results for '{search_keyword}':")
        st.dataframe(filtered_df[['title']])  # title 열만 표시
    else:
        st.write(f"No exact matches found for '{search_keyword}'.")

    # 추천 키워드 찾기 (결과 유무와 관계없이 항상 추천)
    recommendations = difflib.get_close_matches(search_keyword, all_keywords, n=num_recommendations, cutoff=0.5)
    
    if recommendations:
        show_recommendations(recommendations)
    else:
        st.sidebar.write("No similar keywords found.")
else:
    st.sidebar.write("Enter a keyword to search.")

# # 전체 데이터 미리보기 (옵션)
# st.write("Preview of merged_결과.csv (First 5 rows):")
# st.dataframe(df[['title']].head())






# 예시 데이터프레임 생성
data = {
    '이름': ['홍길동', '이몽룡', '성춘향', '장보고', '을지문덕'],
    '직업': ['학생', '학생', '주부', '해적', '장군']
}
df = pd.DataFrame(data)

st.title("다중 필드 검색이 있는 Streamlit 앱")

# 검색창 추가
name_query = st.text_input("이름을 입력하세요:", "")
job_query = st.text_input("직업을 입력하세요:", "")

# 필터링 조건 생성
filtered_df = df.copy()

if name_query:
    filtered_df = filtered_df[filtered_df['이름'].str.contains(name_query)]

if job_query:
    filtered_df = filtered_df[filtered_df['직업'].str.contains(job_query)]

# 검색 결과 표시
if not filtered_df.empty:
    st.write("**검색 결과:**")
    st.dataframe(filtered_df)
    st.bar_chart(filtered_df['직업'].value_counts())
else:
    st.write("검색 결과가 없습니다.")


# 직업별 빈도수 계산
job_counts = df['직업'].value_counts().reset_index()
job_counts.columns = ['직업', '빈도']

# Plotly를 사용한 바 차트 생성 (색상 지정)
fig = px.bar(job_counts, x='직업', y='빈도', 
             color='직업', # 직업별로 다른 색상 적용
             color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']) # 원하는 색상 코드

# Streamlit에 바 차트 표시
st.plotly_chart(fig)

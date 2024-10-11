{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02026ba3-79d1-4ebf-8dfc-00a284cd82ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "\n",
    "# 제목 표시\n",
    "st.title('파이썬 코드 간단 웹 애플리케이션')\n",
    "\n",
    "# 입력을 받는 간단한 코드\n",
    "name = st.text_input('이름을 입력하세요')\n",
    "st.write(f'안녕하세요, {name}!')\n",
    "\n",
    "# 파이썬 코드의 결과를 웹에 출력\n",
    "if st.button('결과 보기'):\n",
    "    result = f\"{name}님을 위한 결과입니다!\"\n",
    "    st.write(result)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

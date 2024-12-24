
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 모델과 토크나이저 로드
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 질문에 대한 답을 생성하는 함수
def answer_question_without_context(question):
    input_ids = tokenizer.encode(question, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Streamlit UI 구성
st.title("Chairman AI")
question = st.text_input("질문을 입력하세요:")

if question:
    answer = answer_question_without_context(question)
    st.write("답변:", answer)

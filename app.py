import streamlit as st
from src.evaluator import Evaluator
from src.evoluter import Evoluter

if 'dataset' not in st.session_state:
    data_dir = "data/dev.txt"
    with open(data_dir, "r") as f:
        data_lines = f.readlines()
    dataset = {line.split("\t")[0]: int(line.split("\t")[1].strip()) for line in data_lines}
    st.session_state.dataset = dataset

st.set_page_config(page_title="LLM prompt evolution", page_icon="🧮")


if 'evaluator' not in st.session_state:
    st.session_state.evaluator = Evaluator()
    st.session_state.evoluter = Evoluter()

prompt1 = st.text_input("Prompt 1", 'Your task is to classify the comment "positive" or "negative. Return label only without any other text.')
if st.button("Evaluate prompt 1"):
    acc1 = Evaluator().batch_predict(ground_truth=st.session_state.dataset, prompt=prompt1)
    st.session_state.acc1 = acc1
if 'acc1' in st.session_state:
    st.write(f"Prompt 1 accuracy: **{st.session_state.acc1}**")

st.divider()
prompt2 = st.text_input("Prompt 2", 'Given a sentence, classify it as either positive or negative sentiment. Return label only without any other text.')
if st.button("Evaluate prompt 2"):
    acc2 = Evaluator().batch_predict(ground_truth=st.session_state.dataset, prompt=prompt2)
    st.session_state.acc2 = acc2
if 'acc2' in st.session_state:
    st.write(f"Prompt 2 accuracy: **{st.session_state.acc2}**")
st.divider()

if st.button("Prompt evolution", type="primary"):
    st.session_state.user = st.session_state.evoluter.evaluation_prompt.replace("<prompt1>", prompt1).replace("<prompt2>", prompt2)
    new_prompt = st.session_state.evoluter.evolution(prompt1, prompt2)
    st.session_state.new_prompt = new_prompt

if st.session_state.get("user") is not None:
    with st.chat_message("user"):
        st.text(st.session_state.user)
if st.session_state.get("new_prompt") is not None:
    with st.chat_message("assistant"):
        st.text(st.session_state.new_prompt)

st.divider()

if st.session_state.get("new_prompt") is not None:
    if st.button("Evaluate new prompt"):
        new = Evoluter.get_final_prompt(st.session_state.new_prompt)
        accnew = Evaluator().batch_predict(ground_truth=st.session_state.dataset, prompt=new)
        st.session_state.accnew = accnew
    if st.session_state.get("accnew") is not None:
        st.write(f"New prompt accuracy: **{st.session_state.accnew}**")

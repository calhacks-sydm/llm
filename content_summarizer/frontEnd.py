from typing import Set
import streamlit as st
from streamlit_chat import message
from summarizer import load_chain

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

#init session states
if ("chat_answers_history" not in st.session_state 
    and "user_prompt_history" not in st.session_state 
    and "chat_history" not in st.session_state
    ):
    st.session_state["model_answer_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

## STREAMLIT COMPONENTS
# header
st.title('Edurite Content Assistant')
st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")
#side bar for model settings
with st.sidebar.expander("🛠️ ", expanded=True):
    # Option to preview memory store
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','gpt-4','text-davinci-003','text-davinci-002','code-davinci-002'])
    K = st.number_input(' Number of retreivals',min_value=1,max_value=5)
    COURSE= st.selectbox(label='Course', options=['CS170', 'CS189','EECS 127'])

# initial text input
input_text = st.text_input("Prompt", placeholder="Enter your message here...") or st.button(
    "Submit"
)
def run_llm(input_text):
   qa= load_chain(MODEL,K)
   return qa({"question": input_text,"chat_history": st.session_state["chat_history"] })
    
#act on user's input
if input_text:
   with st.spinner("Generating response..."):
    generated_response=run_llm(input_text)
    #print(generated_response)
    #generated_response, memory= load_chain(query= input_text, model=MODEL)
    sources = set(
        [(doc.metadata["source"] + ",page "+str(int(doc.metadata["page"]))) for doc in generated_response["source_documents"]]
    )
    formatted_response = (
        f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
    )
    st.session_state.user_prompt_history.append(input_text)
    st.session_state.model_answer_history.append(formatted_response)
    st.session_state.chat_history.append((input_text, generated_response["answer"]))

#populate current conversation
with st.expander("Conversation", expanded=True):
    if st.session_state.model_answer_history:
        for generated_response, user_query in zip(
            st.session_state.model_answer_history,
            st.session_state.user_prompt_history,
        ):
            message(
                user_query,
                is_user=True,
            )
            message(generated_response)
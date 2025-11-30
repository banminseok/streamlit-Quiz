import json 
import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser


# QuizGPT를 구현하되 다음 기능을 추가합니다:

# 함수 호출을 사용합니다.
# 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
# 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
# 만점이면 st.ballons를 사용합니다.
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
# st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)
st.title("QuizGPT")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #FFE08F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


output_parser = JsonOutputFunctionsParser()

function = {
    "name": "create_quiz",
    "description": "A function that takes a list of questions and answers and returns a quiz.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions":{
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string"
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string"
                                    },
                                    "correct": {
                                        "type": "boolean"
                                    }
                                },
                                "required": ["answer", "correct"],
                            }
                        }
                    },
                    "required": ["question", "answers"],
                }
            }
        },
        "required": ["questions"],    
    }
}

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    Randomize the order of the answers for each question.
    DO NOT always place the correct answer in the first position.
    The correct answer should be randomly distributed across the 4 options (A, B, C, D) with equal probability.
    
    The difficulty level of the questions should be: {difficulty}
    Easy: The difficulty level is suitable for elementary school students to answer correctly.
    Hard: The questions should be as difficult as graduate-level exam questions.

    Your turn!
         
    Context: {context}
""",
        ),
    ]
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty, api_key):
    llm = ChatOpenAI(
        temperature=0.1,
        model_name="gpt-4o-mini",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_key=api_key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )
    chain = questions_prompt | llm | output_parser
    return chain.invoke({
        "context": format_docs(_docs),
        "difficulty": difficulty
    })

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, )
    docs = retriever.get_relevant_documents(term)
    return docs


# --sidebar---------------------------------------------------
with st.sidebar:
    docs = None
    apiKey = st.text_input("OPENAI_API_KEY 입력")
    apiKey = apiKey.strip()
    if apiKey:
        
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )
        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)
    
    st.markdown("---")
    st.markdown("[Github Repository](https://github.com/banminseok/streamlit-Quiz)")
# --sidebar---------------------------------------------------

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.

    <hr/>

    “QuizGPT에 오신 것을 환영합니다.

    위키백과 문서나 여러분이 업로드한 파일을 바탕으로 퀴즈를 만들어, 지식을 점검하고 공부를 도와드릴게요.
    
    사이드바에서 파일을 업로드하거나 위키백과를 검색해 시작해 보세요.”
    """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("출제 준비가 되었습니다. <br>", unsafe_allow_html=True)
    choiceLevel = st.selectbox(
        "Please select the difficulty level of the quiz.",
        (
            "난이도를 선택하세요 !!",
            "Easy",
            "Hard",
        ),
    )

    if choiceLevel != "난이도를 선택하세요 !!":
        response = run_quiz_chain(docs, topic if topic else file.name,choiceLevel, apiKey)
        with st.form("questions_form"):
            correct_count = 0
            for idx, question in enumerate(response["questions"]):
                st.write(f"**Q{idx+1}. {question['question']}**")
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=f"q_{idx}"
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                    correct_count += 1
                elif value is not None:
                    st.error("Wrong!")
            
            button = st.form_submit_button("Submit")
            
            if correct_count == len(response["questions"]):
                st.balloons()

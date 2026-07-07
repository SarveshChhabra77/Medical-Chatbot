from flask import Flask, render_template, jsonify, request, session
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from src.prompt import *
import os
import uuid


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key-for-medical-chatbot")


load_dotenv()
 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

os.environ['GROQ_API_KEY'] = GROQ_API_KEY 
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

embeddings = download_embedding()

index_name = 'medial-chatbot'

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=GROQ_API_KEY
)

# Contextualize question prompt for history rephrasing
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# QA prompt template incorporating chat history placeholder
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        MessagesPlaceholder("chat_history"),
        ('human' , '{input}'),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Conversational RAG state management
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


@app.route('/')
def index():
    # Make sure we have a session ID
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template('chat.html')


@app.route('/get',methods=['GET','POST'])
def chat():
    # Ensure session ID is initialized
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    
    session_id = session["session_id"]
    msg = request.form['msg']
    input = msg
    print(f"Session: {session_id} | Input: {input}")
    
    try:
        response = conversational_rag_chain.invoke(
            {"input": input},
            config={"configurable": {"session_id": session_id}}
        )
        print('Response : ', response['answer'])
        return response['answer']
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        return "⚠️ I encountered an error while retrieving answers. Please try again."


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0',port=port)
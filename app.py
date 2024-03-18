from flask import Flask, render_template, request
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

# Set OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "INSERT_OPENAI_KEY"

# Load documents using DirectoryLoader
loader = DirectoryLoader("data", glob="*.txt")
documents = loader.load()

# Split documents using CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=625, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Create Conversation Retrieval Chain
chain = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever())

# Create Conversation Memory
chat_history = []

# Render HTML form for user input
@app.route('/')
def index():
    return render_template('index.html')

# Handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    user_question = request.form['user_question']
    response = answer_question(user_question)
    return render_template('index.html', user_question=user_question, response=response)

# Function to answer questions
def answer_question(question):
    global chat_history
    result = chain({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    return result['answer']

if __name__ == '__main__':
    app.run(debug=True)

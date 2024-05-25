import os
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

# Step 2: Define the BeautifulSoup strainer
bs4_strainer = bs4.SoupStrainer(class_=("mw-page-title-main", "page__main"))

# Step 3: Initialize WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://tardis.fandom.com/wiki/Fifteenth_Doctor","https://tardis.fandom.com/wiki/Fourteenth_Doctor","https://tardis.fandom.com/wiki/Doctor","https://tardis.fandom.com/wiki/Thirteenth_Doctor"),
    bs_kwargs={"parse_only": bs4_strainer},
)
# print(loader)
# Step 4: Load documents
docs = loader.load()
# print(docs)
# Step 5: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)
# print(all_splits)  # Debugging: Check split documents

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Step 6 & 7: Create Chroma object and set up retriever
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Step 8: Retrieve documents
retrieved_docs = retriever.invoke("Who is the doctor?")
print(len(retrieved_docs))

# Step 9: Set up environment variable for OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

# Step 10: Initialize ChatOpenAI with the API key
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=api_key)

# Step 11: Pull a prompt and invoke it with example messages
prompt = hub.pull("rlm/rag-prompt")
example_messages = prompt.invoke({"context": "filler context", "question": "filler question"}).to_messages()
print(example_messages)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
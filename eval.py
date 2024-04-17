from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Loading Model
print('Loading the model...')
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
import openai

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# Loading PDF
print('Loading the corpus for TESLA...')
from langchain.document_loaders import PyPDFLoader
# create a loader
loader = PyPDFLoader(r"data/tsla-20230930.pdf")

# load your data
data = loader.load()

# Text splitter
print('Instantiating Text Splitter...')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=300)
all_splits = text_splitter.split_documents(data)

# Creating Embeddings
print('Preparing Embeddings...')
model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {"device": device}

embeddings = HuggingFaceEmbeddings(model_name=model_name)#, model_kwargs=model_kwargs)

print('Preparing Vector Embeddings...')
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

# Preparing Chain
print('Preparing chain...')

# Define prompt template
template = """You are an assistant for question-answering tasks for Retrieval Augmented Generation system. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use two sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
retriever = vectordb.as_retriever()

# Setup RAG pipeline
conversation_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)


from datasets import Dataset

questions = ["What’s the total assets?",
             "How much cash was provided by or used in operating activities during the quarter?",
             "What are the biggest risks for Tesla as a business?",
            ]
ground_truth = [["The total assets of the company is $93,941."],
                ["The amount of cash provided by or used in operating activities during the quarter was $8.89 billion."],
                ["The biggest risks for Tesla as a business are its ability to continue as a going concern and its inability to raise additional capital to fund its operations and growth."]]
answers = []
contexts = []

# Inference
for query in questions:
  answers.append(conversation_chain.invoke(query))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truth
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()
df.to_csv('rag_gpt35_eval.csv')
print(df)
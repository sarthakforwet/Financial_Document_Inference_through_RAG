from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI


from transformers import LlamaForCausalLM, LlamaTokenizer

# Transformers packages
from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time
#import chromadb
#from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
# from transformers import AutoModelForSeq2SeqLM


# Loading Model
print('Loading the model...')
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-large"
model_config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, config = model_config, trust_remote_code = True, device_map = 'auto')
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Creating Pipeline
print('Creating Pipeline...')
query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",)
llm = HuggingFacePipeline(pipeline=query_pipeline)


# Loading PDF
print('Loading the corpus for TESLA...')
from langchain.document_loaders import PyPDFLoader
# create a loader
loader = PyPDFLoader(r"S:\DS 5983 Final Project\tsla-20230930.pdf")

# load your data
data = loader.load()


# Text splitter
print('Instantiating Text Splitter...')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=300)
all_splits = text_splitter.split_documents(data)

# Creating Embeddings
print('Preparing Embeddings...')
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


print('Preparing Vector Embeddings...')
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

# Preparing Chain
print('Preparing chain...')
conversation = load_qa_chain(llm, chain_type="map_reduce")

print('Chain Prepared...')
# chat = OpenAI(temperature=0)



# conversation = ConversationChain(
#     llm=chat, 
#     verbose=True,
#     memory=ConversationBufferMemory()
# )
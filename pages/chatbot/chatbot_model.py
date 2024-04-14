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

model_id = "google/flan-t5-large"
#model_id = '/kaggle/input/llama-2/pytorch/13b-chat-hf/1'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    load_in_4bit=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                              trust_remote_code=True,
                                              config=model_config,
                                              device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Creating Pipeline
print('Creating Pipeline...')
query_pipeline = transformers.pipeline(
        "text2text-generation",
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
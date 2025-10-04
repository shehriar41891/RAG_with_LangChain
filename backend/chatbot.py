#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import all the necessary packages 
from langchain_core.prompts import PromptTemplate
import os
import pinecone
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from pinecone import ServerlessSpec,Pinecone
# from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import random
index_name = 'pdfreader'


# OpenAI Setup

# In[4]:


load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

print('There is an api ky here')
# In[5]:

print('Small chnage at  the begninig')
#instantiate the openai 
llm = OpenAI(api_key=openai_api_key)

print('Added some chnage here')
# In[6]:


# instantiating the openai embedding
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(api_key=openai_api_key)


# Load the Document

# In[7]:


def load_book(path):
    loader = DirectoryLoader(
        path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    document = loader.load()
    
    return document



# In[9]:

#splitting into chunks 
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    
    text_chunks = text_splitter.split_documents(documents)
    
    return text_chunks


# In[10]:


# text_chunks = split_text(document)
# len(text_chunks)


# PINECONE SETUP

# In[11]:


#extracting the environment variable 
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env_key = os.getenv('PINECONE_ENV_KEY')


# In[14]:

pc = Pinecone(
    api_key=pinecone_api_key
)

def pinecone_connection():
    # random_num = random.randint(1,100)

    # index_name = f'pdfreader{random_num}' #specify the index name where we have stored the embedding

    #if you do not have index you can run this code else skip
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name = index_name,
            dimension=1536,
            spec=ServerlessSpec(
                region=pinecone_env_key,
                cloud='aws'
            )
        )


# In[ ]:


# to create embedding for your text chunks

def create_embedding(queries):
    embedding_list = []
    for query in queries:
        embedding_list.append(embedding.embed_query(query.page_content))
    return embedding_list
    
# for text in text_chunks:
#     create_embedding(text.page_content)

# len(embedding_list)


# In[16]:


import itertools

index_ = pc.Index(index_name)


# In[ ]:

def upsert_data(embedding_list,text_chunks):
    metadata_list = [{"text": text_chunks[i].page_content } for i in range(len(embedding_list))]

    data_to_upsert = [
        {
            'id': f"id-{i}",
            'values': embedding,
            'metadata': metadata_list[i]  # Include metadata
        }
        for i, embedding in enumerate(embedding_list)
    ]
    
    return data_to_upsert


# In[ ]:


def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


# In[ ]:


# # Upsert data with 200 vectors per upsert request
# for ids_vectors_chunk in chunks(data_to_upsert, batch_size=200):
#     index.upsert(vectors=ids_vectors_chunk) 


# # PromptTemplate

# In[18]:


prompt = PromptTemplate(
    input_variables= ['answer','question'],
    template = '''You are given a answer for a question. Refine the given answer based on the question and give relevent answer to 
    the question. Give a general answer 
    answer: {answer},
    question: {question}
    
    Please answer clearly
    '''
)


# def quering(query):
#     query = query
#     vector = embedding.embed_query(query)

#     query_result = index_.query(
#         vector = vector,
#         top_k=3,
#         include_values = True
#     )

#     matched_id = [query_result.matches[0].id]

#     metadata_results = index_.fetch(ids=matched_id)

#     result = ''

#     return result

question = ''
def quering(query):
    question =query
    # Generate the embedding vector for the query
    vector = embedding.embed_query(query)
    
    # Query the Pinecone index to get the top matches
    query_result = index_.query(
        vector=vector,
        top_k=3,
        include_values=True,
        include_metadata=True  # Ensure metadata is included directly in the query results
    )
    
    # Initialize an empty string to hold the result
    result = ''
    
    # Process the query results
    for match in query_result.matches:
        # Retrieve metadata and other information for each match
        matched_id = match.id
        metadata = match.metadata  # Access metadata directly from match
        
        print(metadata,matched_id)
        
        result = metadata['text']
                
        print(result)
    
    return result


# use the langchain to refine the result 

from langchain.chains import LLMChain

def refining_result(result):
    chain = LLMChain(
        prompt = prompt,
        llm = llm
    )

    refined_result = chain.run(answer=result,question=question)
    
    return refined_result


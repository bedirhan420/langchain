from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

'''
YAPI 

documents => context Documentler ile tanımlıyoz

embeddings => hf embedding yüklüyoz (alternatif OpenAIEmbedding)

vectore_store => Chroma ile contexti ve embeddingsi veriyoz

retriver => RunnableLambda ile vector_store ile benzerlik araması yapıyoz 
    bind ile bulduklarından kaç tanesini getirceğini veriyoz (k)

model => transformer pipeline ile modeli oluşturuyoz (HF)

parser => StrOutputParser ile parserı oluşturuyoz bu bize sadece istediğimiz sonucu (content) veriyo

prompt_text => {} lar içinde verdiğimiz input_parameterla prompt_texti giriyoz
    bu openAI tarzı modellerde systempromt ve userprompt diye ikiye ayrılıyo
    
promt_template => input_variable ları ve prompt_texti tanımlıyoz

zincir => LCEL ile | zinciri oluşturuyoz

çalıştırma => chaindeki invoke metoduyla input parametreleri veriyoz

'''

documents = [
    Document(page_content="Dogs are great companions, known for their loyalty and friendliness.", metadata={"source": "mammal-pets-doc"}),
    Document(page_content="Cats are independent pets that often enjoy their own space.", metadata={"source": "mammal-pets-doc"}),
    Document(page_content="Goldfish are popular pets for beginners, requiring relatively simple care.", metadata={"source": "fish-pets-doc"}),
    Document(page_content="Parrots are intelligent birds capable of mimicking human speech.", metadata={"source": "bird-pets-doc"}),
    Document(page_content="Rabbits are social animals that need plenty of space to hop around.", metadata={"source": "mammal-pets-doc"}),
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(documents, embedding=embeddings)

retriver = RunnableLambda(vector_store.similarity_search).bind(k=1)

flan_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device=-1,
    max_length=100
)

model = HuggingFacePipeline(pipeline=flan_pipeline)

str_parser = StrOutputParser()

prompt_text = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt_temp = ChatPromptTemplate.from_messages([
    ("human", prompt_text)
])

rag_chain = {"context":retriver,"question":RunnablePassthrough()} | prompt_temp | model | str_parser

if __name__ == "__main__":
    response = rag_chain.invoke("tell me about cats")
    print(response)

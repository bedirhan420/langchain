from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from transformers import pipeline
import bs4
from dotenv import load_dotenv
load_dotenv()
'''
YAPI 
model => transformer pipeline ile modeli oluşturuyoz (HF)

parser => StrOutputParser ile parserı oluşturuyoz bu bize sadece istediğimiz sonucu (content) veriyo

loader => WebBaseLoader ile web sitesinden verileri çekiyoz
        loader.load()   

embeddings => hf embedding yüklüyoz (alternatif OpenAIEmbedding)
    
text_splitter => RecursiveCharacterTextSplitter ile documenti chunksize ve overlap vererek
    chunklara bölüyoz   

vectore_store => Chroma ile contexti ve embeddingsi veriyoz

retriver => vector_storeu retrivera dönüştüryoz as_retriver

prompt => hubdan rag için yazılmış bi prompt çekiyoz
    
zincir => LCEL ile | zinciri oluşturuyoz

çalıştırma => chaindeki stream metoduyla geldikçe tane tane gösteriyoz
'''

flan_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device=-1,
    max_length=100
)

model = HuggingFacePipeline(pipeline=flan_pipeline)

str_parser = StrOutputParser()

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content","post-title","post-header")
        )
    )
)

docs = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
splits = text_splitter.split_documents(docs)
vector_store = Chroma.from_documents(documents=splits,embedding=embeddings)

retriver = vector_store.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context":retriver | format_docs,"question":RunnablePassthrough()}
    | prompt
    | model
    | str_parser
)

def main():
    for chunk in rag_chain.stream("What is Task Decomposition?"):
        print(chunk,end="",flush=True)


if __name__ == "__main__":
    main()

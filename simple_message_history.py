from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from transformers import pipeline
import uuid
'''
YAPI 

model => transformer pipeline ile modeli oluşturuyoz (HF)

sezon_id => geçmişi kaydetmek için bir değişken ve sezon_id yi tanımlamak için 
    eğer yoksa InMemoryChatMessageHistory ile oluşturuyoz ve dönen tip BaseChatMesaageHsistory
    
promt_template => input_variable ları ve prompt_texti tanımlıyoz 
    MessagePlaceHolder ile historyyi veriyoz

parser => StrOutputParser ile parserı oluşturuyoz bu bize sadece istediğimiz sonucu (content) veriyo

zincir => LCEL ile | zinciri oluşturuyoz

geçmiş zinciri => RunnableWİthMessageHistory ile 
    zinciri , sezonidyi oluşturan fonksiyonu ve parameetreleri giriyoz
'''

flan_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device=-1,
    max_length=500
)

model = HuggingFacePipeline(pipeline=flan_pipeline)

store = {}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
    "You are a helpful AI assistant. When a product review is given, analyze its sentiment (positive, neutral, negative)"
    "If asked about previous reviews, answer accordingly in full sentences.\n"
    "Always respond in this format:\n"
    "Sentiment: <positive/neutral/negative>\n"
    "Explanation: <brief explanation>"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

str_parser = StrOutputParser()

chain = prompt_template | model | str_parser

history_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def interact(session_id: str, user_input: str):
    inputs = {
        "input": user_input
    }
    config = {"configurable": {"session_id": session_id}}
    try:
        result = history_chain.invoke(inputs, config=config)
        return result.strip()
    except Exception as e:
        return f"Analiz hatası: {e}"


def main():
    print("Ürün Yorumu Analiz ve Puanlama")
    print("Çıkmak için 'quit' yazabilirsiniz.\n")
    session_id = str(uuid.uuid4())
    print(f"Oturum ID'niz: {session_id}\n")
    
    while True:
        user_input= input("YOU : ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input :
            print("Input boş olamaz.\n")
            continue
        
        result = interact(session_id,user_input)
        print(f"🤖 Asistan: {result}\n")

if __name__ == "__main__":
    main()

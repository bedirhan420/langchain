from langchain_huggingface.llms import HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()

'''
YAPI 

model => transformer pipeline ile modeli oluşturuyoz (HF)

promt => promptu yazıyoz

çalıştırma => modeldeki invoke metoduyla promptu giriyoz

'''

hf = HuggingFacePipeline.from_model_id(
    model_id="Helsinki-NLP/opus-mt-en-it",
    task="translation",  
    pipeline_kwargs={"max_new_tokens": 50}
)

prompt = "translate English to Italian: Hi!"
response = hf.invoke(prompt)
print(response)

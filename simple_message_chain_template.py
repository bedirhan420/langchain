from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

'''
YAPI 

prompt_text => {} lar içinde verdiğimiz input_parameterla prompt_texti giriyoz
        bu openAI tarzı modellerde systempromt ve userprompt diye ikiye ayrılıyo
    
promt_template => input_variable ları ve prompt_texti tanımlıyoz

parser => StrOutputParser ile parserı oluşturuyoz bu bize sadece istediğimiz sonucu (content) veriyo

model => transformer pipeline ile modeli oluşturuyoz (HF)

zincir => LCEL ile | zinciri oluşturuyoz

çalıştırma => chaindeki invoke metoduyla input parametreleri veriyoz

'''


# Prompt template (input_variables: review_text, product_name)
prompt_text = (
    "You are a product review assistant.\n"
    "Product: {product_name}\n"
    "Review: {review_text}\n\n"
    "Please answer these questions briefly:\n"
    "1. What is the overall sentiment of the review? (positive, negative, neutral)\n"
    "2. What rating would you give this product out of 5 based on this review?\n\n"
    "Answer format:\n"
    "Sentiment: <sentiment>\n"
    "Rating: <rating>"
)

prompt_template = PromptTemplate(
    input_variables=["product_name", "review_text"],
    template=prompt_text
)

str_parser = StrOutputParser()

flan_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device=-1,
    max_length=100
)

model = HuggingFacePipeline(pipeline=flan_pipeline)

chain = prompt_template | model | str_parser

def analyze_review(product_name, review_text):
    inputs = {
        "product_name": product_name,
        "review_text": review_text
    }
    try:
        result = chain.invoke(inputs)
        return result.strip()
    except Exception as e:
        return f"Analiz hatası: {e}"

def main():
    print("Ürün Yorumu Analiz ve Puanlama")
    print("Çıkmak için 'quit' yazabilirsiniz.\n")
    
    while True:
        product_name = input("Ürün adı: ").strip()
        if product_name.lower() == "quit":
            break
        review_text = input("Ürün yorumu: ").strip()
        if review_text.lower() == "quit":
            break
        if not product_name or not review_text:
            print("Her iki alan da boş olamaz.\n")
            continue
        
        result = analyze_review(product_name, review_text)
        print("\n--- Analiz Sonucu ---")
        print(result)
        print("---------------------\n")

if __name__ == "__main__":
    main()

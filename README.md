conda create -n langchain python=3.12.11
conda activate langchain
pip install -r requirements.txt

-----------------------------------

# LANGCHAIN

## `langchain`

### `langchain.prompts`

- **`PromptTemplate`**: Düz metin temelli prompt şablonları.
    
    ```python
    from langchain.prompts import PromptTemplate
    
    prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    print(prompt.format(product="smartphone"))
    ```
    
- **`ChatPromptTemplate`**: Çoklu mesaj içeren chat temelli prompt yapısı.
    
    ```python
    from langchain.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "What is the capital of {country}?")
    ])
    print(prompt.format_messages(country="Germany"))
    ```
    
- **`FewShotPromptTemplate`**: Az örnekli öğrenme için örnekleri içeren şablon.
    
    ```python
    from langchain.prompts import FewShotPromptTemplate, PromptTemplate
    
    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "3*3", "output": "9"}
    ]
    
    example_prompt = PromptTemplate(input_variables=["input", "output"], template="Q: {input} A: {output}")
    
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Q: {input} A:",
        input_variables=["input"]
    )
    
    print(prompt.format(input="5-2"))
    ```
    
- **`MessagesPlaceholder`**: Geçici mesaj bölmesi (örneğin geçmiş mesajlar için).
    
    ```python
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "What's the weather today?")
    ])
    
    # Kullanım senaryosu: Agent geçmişi aktarırken doldurur.
    formatted = prompt.format_messages(chat_history=[
        ("human", "Hello"),
        ("ai", "Hi there!")
    ])
    print(formatted)
    ```
    

### `langchain.agents`

- **`AgentExecutor`**: Ajanları çalıştırmak için kullanılan ana sınıf.
- **`initialize_agent()`**: Aracılar ve LLM ile bir agent başlatır.
- **`Tool`**: Ajanların kullanabileceği bağımsız işlevsel modüller.
- **`AgentType`**: Kullanılabilir ajan türleri (zero-shot, self-ask vb.).
    
    ```python
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_openai import ChatOpenAI
    from langchain import hub
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = [TavilySearchResults(max_results=1)]
    prompt = hub.pull("hwchase17/react")  # ReAct template
    
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    result = executor.invoke({"input": "What's the tallest building in Turkey?"})
    print(result)
    ```
    

### `langchain.chains`

- **`LLMChain`**: Prompt + LLM ile en basit zincir.
    
    ```python
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_openai import OpenAI
    
    llm = OpenAI()
    prompt = PromptTemplate.from_template("Translate '{sentence}' to French.")
    chain = LLMChain(llm=llm, prompt=prompt)
    
    print(chain.run("I love apples."))
    ```
    
- **`ConversationalRetrievalChain`**: Chat tabanlı RAG (soru-cevap).
    
    ```python
    from langchain.chains import ConversationalRetrievalChain
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    
    # Belgeleri hazırla
    docs = TextLoader("data.txt").load()
    chunks = CharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding)
    
    # Chain'i kur
    llm = ChatOpenAI()
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    
    chat_history = []
    query = "Who is Albert Einstein?"
    response = qa_chain.run({"question": query, "chat_history": chat_history})
    print(response)
    ```
    
- **`RetrievalQA`**: Retriever ile LLM çıktısını birleştirerek cevap üretir.
    
    ```python
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    
    llm = ChatOpenAI()
    retriever = FAISS.load_local("my_faiss_index", OpenAIEmbeddings()).as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    print(qa.run("Explain LangChain in simple terms.")
    ```
    

---

## `langchain_core`

### `langchain_core.messages`

- **`HumanMessage`**, **`AIMessage`**, **`SystemMessage`**: Chat geçmişini temsil eden mesaj türleri.
    
    ```python
    from langchain_core.messages import HumanMessage, AIMessage
    
    messages = [
        HumanMessage(content="Hello, who won the World Cup in 2018?"),
        AIMessage(content="France won the World Cup in 2018.")
    ]
    ```
    
- **`FunctionMessage`**: OpenAI function calling için çıktı mesajı.
- **`ToolMessage`**, **`ToolCall`**: Agent aracılar için tool kullanımı mesajları.

### `langchain_core.prompts`

- **`PromptTemplate`**, **`ChatPromptTemplate`**: Soyutlanmış prompt sınıfları.
- **`format()`**, **`partial()`** gibi fonksiyonlarla esnek prompt üretimi yapılır.
    
    ```python
    from langchain_core.prompts import PromptTemplate
    
    template = PromptTemplate.from_template("Translate '{sentence}' to French.")
    print(template.format(sentence="How are you?"))
    ```
    

### `langchain_core.output_parsers`

- **`StrOutputParser`**: Model çıktısını düz string olarak döner.
- **`JsonOutputParser`**: JSON çıktısını ayrıştırır.
- **`PydanticOutputParser`**: Pydantic modellerine dönüştürür.
    
    ```python
    from langchain_core.output_parsers import StrOutputParser
    
    parser = StrOutputParser()
    output = parser.invoke("Hello world") 
    print(output)
    ```
    

### `langchain_core.runnables`

- **`Runnable`, `RunnableLambda`, `RunnableMap`**: Zincir parçalarını fonksiyonel olarak bağlamaya yarar.
- **`invoke()`, `stream()`** gibi metodlarla çalıştırılır.
    
    ```python
    from langchain_core.runnables import RunnableLambda
    
    uppercase = RunnableLambda(lambda x: x.upper())
    print(uppercase.invoke("langchain"))  # Output: LANGCHAIN
    ```
    

### `langchain_core.tools`

- **`BaseTool`**: Tüm araç sınıfları için temel yapı.
- **`tool()` decorator**: Python fonksiyonunu araca dönüştürür.
    
    ```python
    from langchain_core.tools import tool
    
    @tool
    def multiply(x: int, y: int) -> int:
        """İki sayıyı çarp"""
        return x * y
    
    print(multiply.invoke({"x": 3, "y": 4}))  # 12
    ```
    

---

## `langchain_community`

### `langchain_community.tools`

- Arama araçları (Tavily, SerpAPI), hesaplayıcılar, Python çalıştırıcılar.
- **`TavilySearchResults`**, **`DuckDuckGoSearchResults`**
    
    ```python
    from langchain_community.tools.tavily_search import TavilySearchResults
    
    search_tool = TavilySearchResults()
    print(search_tool.invoke({"query": "current president of France"}))
    ```
    

### `langchain_community.vectorstores`

- FAISS, Chroma, Pinecone, Weaviate gibi vektör veri tabanları.
- **`FAISS`, `Chroma`, `Weaviate`, `Pinecone`**
    
    ```python
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    
    texts = ["LangChain is powerful", "OpenAI builds AI", "FAISS is fast"]
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embedding=embedding)
    
    docs = vectorstore.similarity_search("AI development", k=1)
    print(docs[0].page_content)
    ```
    

### `langchain_community.document_loaders`

- PDF, web, txt, CSV gibi belgeleri yükler.
- **`WebBaseLoader`, `PyPDFLoader`, `TextLoader`**
    
    ```python
    from langchain_community.document_loaders import WebBaseLoader
    
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Natural_language_processing")
    docs = loader.load()
    print(docs[0].page_content[:500])
    ```
    

### `langchain_community.embeddings`

- HuggingFace, OpenAI, Cohere embed modelleri.
- **`HuggingFaceEmbeddings`, `OpenAIEmbeddings`**

### `langchain_community.llms` ve `chat_models`

- HF, OpenAI, Replicate, Cohere gibi LLM'ler.
- **`HuggingFaceHub`, `ChatOpenAI`, `ChatAnthropic`**

---

## `langchain_huggingface`

### `langchain_huggingface.llms`

- **`HuggingFacePipeline`**: Yerel olarak çalışan `transformers.pipeline` nesnesiyle LLM bağlama.
- **`HuggingFaceEndpoint`**: Hugging Face Inference API üzerinden model çağırma.
    
    ```python
    from langchain_huggingface.llms import HuggingFacePipeline
    from transformers import pipeline
    
    flan_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        tokenizer="google/flan-t5-large",
        device=-1,
        max_length=100
    )
    
    model = HuggingFacePipeline(pipeline=flan_pipeline)
    ```
    

---

## `langchain_google_genai`

### `langchain_google_genai`

- **`ChatGoogleGenerativeAI`**: Google Gemini modelleri için chat arayüzü.
- **`GoogleGenerativeAIEmbeddings`**: Embedding işlemleri için.
- **`GoogleGenerativeAI`**: Normal LLM istemcisi (chat olmayan modeller için).
    
    ```python
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    response = llm.invoke("What is quantum computing?")
    print(response.content)
    ```
    

> Google API key gerektirir: env GOOGLE_API_KEY veya servis hesabı JSON'u.
> 

---

## `langchain_openai`

### `langchain_openai`

- **`ChatOpenAI`**: OpenAI GPT modelleri (chat-tabanlı).
- **`OpenAIEmbeddings`**: Text embedding üretimi.
- **`OpenAI`**: Düz metin tamamlama modeli (completion).
- **`OpenAIStream`**: Streaming destekli çıktı.

> 📌 API anahtarı gerekir: .env içinde OPENAI_API_KEY
>
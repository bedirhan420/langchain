from transformers import pipeline
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

str_parser = StrOutputParser()

memory = SqliteSaver.from_conn_string(":memory:")
search = TavilySearchResults(max_results=2)

tools = [search]

agent_exe = create_react_agent(model,tools,checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

if __name__ == "__main__":
    while True:
        user = input("YOU : ")
        if user.strip().lower() in {"q","quit","exit"}:
            break
        agent_input = {"messages":[HumanMessage(content=user)]}
        for chunk in agent_exe.stream(agent_input,config):
            if isinstance(chunk,dict) and "agent" in chunk:
                for msg in chunk["agent"]["messages"]:
                    if isinstance(msg,AIMessage):
                        print("AI: " , msg.content.strip())
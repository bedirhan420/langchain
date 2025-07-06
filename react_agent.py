from transformers import pipeline
from langchain import hub
from langchain.agents import AgentExecutor,create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

memory = SqliteSaver.from_conn_string(":memory:")
search = TavilySearchResults(max_results=2)

tools = [search]

prompt = hub.pull("hwchase17/react-chat")

agent = create_react_agent(model,tools,prompt)
agent_exe = AgentExecutor(agent=agent,tools=tools,verbose=True,checkpoint=memory)
config = {"configurable": {"thread_id": "abc123"}}

if __name__ == "__main__":
    chat_hist = []
    while True:
        user = input("YOU : ")
        if user.strip().lower() in {"q","quit","exit"}:
            break
        chat_hist.append(f"User: {user}")
        response = []
        for chunk in agent_exe.stream({"input":user,
                                       "chat_history":"\n".join(chat_hist)},config):
            if isinstance(chunk,dict) and "agent" in chunk:
                for msg in chunk["agent"]["messages"]:
                    if isinstance(msg,AIMessage):
                        content =  msg.content.strip()
                        print("AI: " ,content)
                        response.append(content)
        if response:
            chat_hist.append(f"AI: {''.join(response)}")
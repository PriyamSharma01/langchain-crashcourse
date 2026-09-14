from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

load_dotenv()
tavily = TavilyClient()

@tool
def search_tool(query: str) -> str:
    """Tools that search the web for relevant information.
    Args:
        query (str): The query to search for.
    Returns:
        str: The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-4o",temperature=0)
tools = [search_tool]
agent = create_agent(llm, tools=tools)




def main():
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in Tokyo?")})
    print(result)

if __name__ == "__main__":
    main()
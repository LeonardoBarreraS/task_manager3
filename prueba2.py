from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.redis import RedisSaver
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Define a simple tool
@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")

# Set up model and tools
tools = [get_weather]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create Redis persistence
REDIS_URI = os.getenv("REDIS_URI")
with RedisSaver.from_conn_string(REDIS_URI) as checkpointer:
    # Initialize Redis indices (only needed once)
    checkpointer.setup()
    
    # Create agent with memory
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    
    # Use the agent with a specific thread ID to maintain conversation state
    config = {"configurable": {"thread_id": "user122"}}
    res = graph.invoke({"messages": [("human", "what's the weather in nyc")]}, config)

    print(res["messages"][-1].content)  # Output: It's always sunny in sf
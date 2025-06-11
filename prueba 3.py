#%%
import getpass
import os
from dotenv import load_dotenv

load_dotenv()

open_ai_key= os.getenv("OPENAI_API_KEY")
REDIS_URI= os.getenv("REDIS_URI")

#%%

from langchain_openai import OpenAIEmbeddings
from langgraph.store.redis import RedisStore
from langgraph.store.base import IndexConfig


# Create index configuration for vector search
index_config: IndexConfig = {
    "dims": 1536,
    "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
    "ann_index_config": {
        "vector_type": "vector",
    },
    "distance_type": "cosine",
}
#%%
# Initialize the Redis store
redis_store = None
with RedisStore.from_conn_string(REDIS_URI, index=index_config) as s:
    s.setup()
    redis_store = s

#%%

import uuid
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.base import BaseStore


model = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        timeout=30,  # 30 second timeout for Railway
        max_retries=2,  # Fewer retries for faster failure detection
        request_timeout=30  # Request-specific timeout
    )


# NOTE: we're passing the Store param to the node --
# this is the Store we compile the graph with
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # Store new memories if the user asks the model to remember
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = "User name is Bob"
        store.put(namespace, str(uuid.uuid4()), {"data": memory})

    response = model.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

# Set up Redis connection for checkpointer
checkpointer = None
with RedisSaver.from_conn_string(REDIS_URI) as cp:
    cp.setup()
    checkpointer = cp

# NOTE: we're passing the store object here when compiling the graph
graph = builder.compile(checkpointer=checkpointer, store=redis_store)
# %%
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
input_message = {"role": "user", "content": "Hi! Remember: my name is Bob"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
# %%

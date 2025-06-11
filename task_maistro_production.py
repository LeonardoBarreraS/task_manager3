# ---------------------------------------------
# Task Maistro Assistant - Persistencia Railway
# Arquitectura: Estado temporal en memoria (MemorySaver), datos persistentes en Postgres (PostgresStore)
# No se usa Redis ni ningún otro checkpointer persistente
# ---------------------------------------------

import uuid
import os
from datetime import datetime
import json


# Core imports with error handling
from pydantic import BaseModel, Field
from trustcall import create_extractor
from typing import Literal, Optional, TypedDict
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore



from langgraph.checkpoint.redis import RedisCheckpoint
from langgraph.store.redis import RedisStore




from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
import configuration


## Schema definitions##

# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has", 
        default_factory=list
    )

# ToDo schema
class ToDo(BaseModel):
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(description="Estimated time to complete the task (minutes).")
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by (if applicable)",
        default=None
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        min_items=1,
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started"
    )

## Initialize the model and tools

# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'todo', 'instructions']


# Initialize the model - lazy loading to ensure API key is available
def get_model():
    """Get ChatOpenAI model with proper error handling and Railway-specific timeouts"""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Warning: OPENAI_API_KEY is not set. OpenAI calls may fail.")
    # Railway-specific configuration with timeouts to prevent hanging
    return ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        timeout=30,  # 30 second timeout for Railway
        max_retries=2,  # Fewer retries for faster failure detection
        request_timeout=30  # Request-specific timeout
    )

model = get_model()

## Create the Trustcall extractors for updating the user profile and ToDo list
profile_extractor= create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )

## Prompts 

# Chatbot instruction for choosing what to update and what tools to call 
MODEL_SYSTEM_MESSAGE = """{task_maistro_role} 

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you update the todo list
- Do not tell the user that you have updated instructions

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# Instructions for updating the ToDo list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""


#########################################################################################################################################
## Node definitions

def task_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memories from the store and use them to personalize the chatbot's response."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category
    task_maistro_role = configurable.task_maistro_role

    user_profile = None
    todo = ""
    instructions = ""

#############################################################################
    # Usar 'store' dentro de un 'with' block para acceder a los métodos
    with store as active_store:
   # Retrieve profile memory from the store
        namespace = ("profile", todo_category, user_id)
        memories = active_store.search(namespace)

        if memories:
            profile_data = memories[0].value
            if isinstance(profile_data, str): # Si se serializó como cadena, deserealizar
                profile_data = json.loads(profile_data)
            user_profile = Profile.model_validate(profile_data).model_dump_json(indent=2)
        else:
            user_profile = None
##################################################################################

        # Retrieve people memory from the store
        namespace = ("todo", todo_category, user_id)
        memories = active_store.search(namespace)
    

        todo_list_formatted = []
        if memories:
            for mem in memories:
                todo_data = mem.value
                if isinstance(todo_data, str):
                    todo_data = json.loads(todo_data)
                todo_list_formatted.append(json.dumps(todo_data))
        todo = "\n".join(todo_list_formatted)

##################################################################################

          # Retrieve custom instructions
        namespace = ("instructions", todo_category, user_id)
        memories = active_store.search(namespace)
        if memories:
            instructions_data = memories[0].value
            if isinstance(instructions_data, str):
                # Las instrucciones pueden ser una cadena simple
                instructions = instructions_data
            else: # Si se guardó como JSON, convertir a cadena.
                instructions = json.dumps(instructions_data)
        else:
            instructions = ""

##############################################################################

    system_msg = MODEL_SYSTEM_MESSAGE.format(task_maistro_role=task_maistro_role, user_profile=user_profile, todo=todo, instructions=instructions)

    # Respond using memory as well as the chat history
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": [response]}




########################################################################################################################################

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category
    # Define the namespace for the memories
    namespace = ("profile", todo_category, user_id)

######################################################################
 

    with store as active_store:

        # Retrieve the most recent memories for context
        existing_items = active_store.search(namespace)

        # Format the existing memories for the Trustcall extractor
        tool_name = "Profile"
        existing_memories = ([(existing_item.key, tool_name, json.loads(existing_item.value) if isinstance(existing_item.value, str) else existing_item.value)
                            for existing_item in existing_items]
                            if existing_items
                            else None
                            )

################################################################################
        # Merge the chat history and the instruction
        TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))    # Invoke the extractor
        result = profile_extractor.invoke({"messages": updated_messages, 
                                            "existing": existing_memories})

##################################################################################

        # Save save the memories from Trustcall to the store
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            active_store.put(namespace, user_id, r.model_dump(mode="json"))


    tool_calls = state['messages'][-1].tool_calls
    # Return tool message with update verification
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0]['id']}]}

####################################################################################################################################

def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    # Define the namespace for the memories
    namespace = ("todo", todo_category, user_id)


##################################################################################

    with store as active_store:

        # Retrieve the most recent memories for context
        existing_items = active_store.search(namespace)

        # Format the existing memories for the Trustcall extractor
        tool_name = "ToDo"
        existing_memories = ([(existing_item.key, tool_name, json.loads(existing_item.value) if isinstance(existing_item.value, str) else existing_item.value)
                            for existing_item in existing_items]
                            if existing_items
                            else None
                            )
        
##################################################################################
        # Merge the chat history and the instruction
        TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

        # Create the Trustcall extractor for updating the ToDo list 
        todo_extractor = create_extractor(
            model,
            tools=[ToDo],
            tool_choice=tool_name,
            enable_inserts=True
        )

        # Invoke the extractor
        result = todo_extractor.invoke({"messages": updated_messages, 
                                            "existing": existing_memories})


############################################################################################

        # Save save the memories from Trustcall to the store
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            todo_id = rmeta.get("json_doc_id", str(uuid.uuid4()))
            active_store.put(namespace, todo_id, r.model_dump(mode="json")) 

    ################################################################################################# 

    # Respond to the tool call made in task_mAIstro, confirming the update    
    tool_calls = state['messages'][-1].tool_calls

    # Extract the changes made by Trustcall and add the the ToolMessage returned to task_mAIstro
    todo_update_msg = "Updated ToDo list:\n"
    return {"messages": [{"role": "tool", "content": todo_update_msg, "tool_call_id":tool_calls[0]['id']}]}

#########################################################################################################################################
def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category
    
    namespace = ("instructions", todo_category, user_id)

############################################################################################################3

    with store as active_store:

        existing_memory_item = active_store.get(namespace, "user_instructions")
        existing_instructions = existing_memory_item.value if existing_memory_item else None
        if existing_instructions and isinstance(existing_instructions, str):
            existing_instructions = json.loads(existing_instructions) # Si se guardó como JSON


#####################################################################################################################


        # Format the memory in the system prompt
        system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_instructions if existing_instructions else "")
        new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'][:-1] + [HumanMessage(content="Please update the instructions based on the conversation")])


###########################################################################################################

        # Overwrite the existing memory in the store 
        key = "user_instructions"
        active_store.put(namespace, key, new_memory.content)

#########################################################################################################

    tool_calls = state['messages'][-1].tool_calls
    # Return tool message with update verification
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id":tool_calls[0]['id']}]}


###########################################################################################################################################
# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) ==0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile"
        elif tool_call['args']['update_type'] == "todo":
            return "update_todos"
        elif tool_call['args']['update_type'] == "instructions":
            return "update_instructions"
        else:
            raise ValueError
#######################################################################################################################################3
## Memory and Store Configuration

def get_checkpointer():
    """Usa RedisCheckpoint para el checkpointer (estado de conversación temporal del grafo)"""
    redis_url = os.getenv("REDIS_URI") # Usaremos la variable de entorno REDIS_URI


    if redis_url:
        print(f"🔴 Using Redis as checkpointer: {redis_url}")
        return RedisCheckpoint(url=redis_url)
    else:
        print("⚠️ REDIS_URI no está configurado. Usando MemorySaver para checkpointer.")
        return MemorySaver() # Fallback si no hay Redis URI

##################################################################################################################

def get_store():
    
    redis_url = os.getenv("REDIS_URI") # Mismo REDIS_URI que para el checkpointer

    if redis_url:
        print(f"📦 Using RedisStore for custom data persistence: {redis_url}")
        return RedisStore(url=redis_url)
    else:
        print("⚠️ REDIS_URI no está configurado. Usando InMemoryStore para datos personalizados.")
        return InMemoryStore() 

#######################################################################################################

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Define the flow of the memory extraction process
builder.add_node(task_mAIstro)
builder.add_node(update_todos)
builder.add_node(update_profile)
builder.add_node(update_instructions)

# Define the flow 
builder.add_edge(START, "task_mAIstro")
builder.add_conditional_edges("task_mAIstro", route_message)
builder.add_edge("update_todos", "task_mAIstro")
builder.add_edge("update_profile", "task_mAIstro")
builder.add_edge("update_instructions", "task_mAIstro")

# Compile the graph with persistent storage 
checkpointer = get_checkpointer()  # Estado de conversación temporal (MemorySaver)
store = get_store()                # Datos persistentes en PostgreSQL (o memoria)

graph = builder.compile(checkpointer=checkpointer, store=store)

# Export graph for use in other modules
__all__ = ["graph"]

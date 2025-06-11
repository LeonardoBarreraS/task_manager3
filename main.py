#!/usr/bin/env python3
"""
Task Maistro Assistant - CLEAN DEPLOYMENT v3.0.0
Railway deployment with threading timeout fixes
"""
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


from task_maistro_production import graph as compiled_graph
from langchain_core.messages import HumanMessage
import time
    
# The graph is already compiled with stable in-memory backends in task_maistro_production.py
print("✅ Graph imported successfully")
print(f"🔍 Graph type: {type(compiled_graph)}")
print("🚀 Using pre-compiled graph with stable in-memory backends...")
print("Graph ready!")
    

def chat_with_assistant(message, history):
    

    try:
        # Check if OpenAI API key is available
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return "Error: OPENAI_API_KEY no está configurada. Por favor, configura la variable de entorno."
        
        # Create config with default values
        config = {
            "configurable": {
                "user_id": "default-user",
                "todo_category": "general", 
                "task_maistro_role": "You are a helpful task management assistant. You help you create, organize, and manage the user's ToDo list."
            },
            "thread_id": "default-thread"
        }
        
        # Create the input message
        input_message = {"messages": [HumanMessage(content=message)]}

        response = compiled_graph.invoke(input_message, config=config)

        # Extract the assistant's response
        assistant_message = response["messages"][-1].content

        return assistant_message
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Application error: {error_msg}")
        return error_msg


def clear_chat():
    """Clear the chat history"""
    return []

# Create the Gradio interface
with gr.Blocks(title="Task Maistro Assistant", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🤖 Task Maistro Assistant")
    gr.Markdown("""
    Tu asistente personal para gestionar tareas y recordatorios. Comparte tus tareas conmigo y te ayudaré a organizarlas.
    
    **🏗️ Arquitectura:**
    - 🔴 **Redis**: Estado de conversación (checkpointer) y datos persistentes (store)
    - 🧠 **LangGraph**: Motor de inteligencia artificial
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=500,
                placeholder="Hola! Soy tu asistente de tareas. ¿En qué puedo ayudarte hoy?",
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Escribe tu mensaje aquí...",
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("Enviar", variant="primary", scale=1)
                clear_btn = gr.Button("Limpiar", variant="secondary", scale=1)
    
    # Event handlers
    def respond(message, history):
        if message.strip() == "":
            return history, ""
        
        # Get response from assistant
        bot_response = chat_with_assistant(message, history)
        
        # Add to history
        history.append([message, bot_response])
        
        return history, ""
    
    # Bind events
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    send_btn.click(respond, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_chat, None, chatbot)


if __name__ == "__main__":
    # Get port from environment variable (Railway sets this)
    port = int(os.getenv("PORT", 8080))  # Changed default to match Dockerfile
    
    # Determine if we're in production (Railway) or development
    is_production = os.getenv("RAILWAY_ENVIRONMENT") is not None
    server_name = "0.0.0.0" if is_production else "127.0.0.1"
    
    # Launch the app
    print(f"Starting application on port {port}")
    print(f"Environment: {'Production (Railway)' if is_production else 'Development (Local)'}")
    print(f"🌐 Access the app at: http://{'0.0.0.0' if is_production else 'localhost'}:{port}")
    
    app.launch(
        server_name=server_name,
        server_port=port,
        share=False,
        show_error=True,
        inbrowser=not is_production,  # Don't auto-open browser in production
        quiet=is_production  # Reduce logging in production
    )

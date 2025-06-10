# 🤖 Task Maistro - Railway Deployment

Un asistente inteligente para gestión de tareas construido con LangChain/LangGraph y desplegado en Railway.

## 🏗️ Arquitectura

- **🐘 PostgreSQL**: Base de datos principal para checkpointer (conversaciones) y store (datos persistentes)
- **🧠 LangGraph**: Motor de inteligencia artificial con memoria persistente
- **🎨 Gradio**: Interface web moderna y responsiva
- **🚀 Railway**: Plataforma de deployment con base de datos integrada

## 🚀 Deploy en Railway

### Prerrequisitos
1. Cuenta en [Railway](https://railway.app)
2. Cuenta en [OpenAI](https://openai.com) con API key

### Pasos para el deployment

1. **Fork/Clone este repositorio**

2. **Conectar con Railway**
   - Ve a [Railway](https://railway.app)
   - Haz clic en "New Project"
   - Selecciona "Deploy from GitHub repo"
   - Conecta este repositorio

3. **Agregar PostgreSQL**
   - En tu proyecto de Railway, haz clic en "New"
   - Selecciona "Database" → "PostgreSQL"
   - Railway creará automáticamente la variable `DATABASE_URL`

4. **Configurar variables de entorno**
   - Ve a la pestaña "Variables" de tu servicio
   - Agrega las siguientes variables:

   ```
   OPENAI_API_KEY=tu_api_key_de_openai
   ANTHROPIC_API_KEY=tu_api_key_de_anthropic (opcional)
   GOOGLE_API_KEY=tu_api_key_de_google (opcional)
   TAVILY_API_KEY=tu_api_key_de_tavily (opcional)
   LANGSMITH_API_KEY=tu_api_key_de_langsmith (opcional)
   ```

5. **Deploy automático**
   - Railway detectará el `Dockerfile` y comenzará el deployment
   - El proceso tomará unos 3-5 minutos

6. **Acceder a tu aplicación**
   - Una vez completado, Railway te dará una URL
   - ¡Tu Task Maistro estará disponible!

## 🔧 Configuración local

1. **Crear entorno virtual**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar variables de entorno**
   ```bash
   cp .env.template .env
   # Edita .env con tus API keys
   ```

4. **Ejecutar localmente**
   ```bash
   python app.py
   ```

## 📊 Funcionalidades

- ✅ **Gestión de tareas**: Crear, organizar y gestionar tareas
- ✅ **Memoria persistente**: Las conversaciones se guardan en PostgreSQL
- ✅ **Interface moderna**: UI intuitiva con Gradio
- ✅ **Múltiples modelos**: Soporte para OpenAI, Anthropic, etc.
- ✅ **Escalable**: Desplegado en Railway con auto-scaling

## 🛠️ Tecnologías

- **Python 3.11+**
- **LangChain & LangGraph**: Framework de IA
- **PostgreSQL**: Base de datos
- **Gradio**: Interface web
- **Railway**: Platform as a Service
- **Docker**: Containerización

## 📝 Notas

- La aplicación usa almacenamiento en memoria cuando no hay `DATABASE_URL`
- En producción con PostgreSQL, todas las conversaciones persisten
- Railway proporciona HTTPS automáticamente
- La aplicación se auto-escala según la demanda

## 🔗 Enlaces útiles

- [Railway Docs](https://docs.railway.app/)
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Gradio Docs](https://gradio.app/docs/)

---

¡Disfruta tu Task Maistro desplegado! 🚀

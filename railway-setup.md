# Railway Setup Guide - Task Maistro

## Required Services in Railway

### 1. PostgreSQL Database
- **Purpose**: Persistent data storage (todos, profiles, instructions)
- **Variable**: `DATABASE_URL`
- **Service**: Add PostgreSQL service in Railway dashboard
- Railway will automatically provide the connection string

### 2. Redis Database  
- **Purpose**: Conversation state checkpointing
- **Variable**: `REDIS_URL`
- **Service**: Add Redis service in Railway dashboard
- Railway will automatically provide the connection string

### 3. Environment Variables Required

```bash
# OpenAI (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Database connections (Auto-provided by Railway)
DATABASE_URL=<auto-provided-by-railway>
REDIS_URL=<auto-provided-by-railway>

# Optional: Port (Railway sets automatically)
PORT=<auto-provided-by-railway>
```

## Deployment Steps

1. **Create Railway Project**
   - Create new project in Railway dashboard
   - Connect your GitHub repository

2. **Add PostgreSQL Service**
   - Click "New Service" → "Database" → "PostgreSQL"
   - Railway will create `DATABASE_URL` automatically

3. **Add Redis Service**
   - Click "New Service" → "Database" → "Redis"
   - Railway will create `REDIS_URL` automatically

4. **Set Environment Variables**
   - Go to your app service → "Variables"
   - Add `OPENAI_API_KEY` with your API key
   - Other variables are auto-provided by Railway

5. **Deploy**
   - Push your code to GitHub
   - Railway will automatically build and deploy

## Architecture

```
🌐 Railway App Service
├── 🐘 PostgreSQL (DATA STORAGE)
│   ├── User profiles
│   ├── Todo lists  
│   └── Instructions
├── 🔴 Redis (CONVERSATION STATE)
│   └── Chat checkpoints
└── 🤖 Python App
    ├── Gradio UI
    └── LangGraph Engine
```

## Memory Configuration

- **Store**: PostgreSQL (`get_store()`)
  - Persistent data that survives restarts
  - User profiles, todos, instructions
  
- **Checkpointer**: Redis (`get_checkpointer()`)
  - Conversation state and history
  - Fast access for chat continuity

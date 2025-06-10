# Task Maistro Assistant - Railway Deployment with Persistent Memory

## 🎯 **Nueva Configuración Implementada**

### **Arquitectura de Memoria:**
- 🔴 **Redis**: Checkpointer para estado de conversación (REAL Redis checkpointer)
- 🐘 **PostgreSQL**: Store para datos persistentes (todos, perfiles, instrucciones)
- 💾 **Fallbacks**: MemorySaver e InMemoryStore si las conexiones fallan

### **Variables de Entorno Necesarias:**
```bash
# OpenAI (Requerido)
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL (Auto-proporcionado por Railway)
DATABASE_URL=postgresql://user:password@host:port/database

# Redis (Auto-proporcionado por Railway)  
REDIS_URL=redis://user:password@host:port
```

## 🚀 **Pasos de Despliegue en Railway**

### 1. **Crear Proyecto Railway**
- Crear nuevo proyecto en Railway dashboard
- Conectar repositorio GitHub

### 2. **Agregar Servicio PostgreSQL**
- Click "New Service" → "Database" → "PostgreSQL"
- Railway creará automáticamente `DATABASE_URL`

### 3. **Agregar Servicio Redis**
- Click "New Service" → "Database" → "Redis"  
- Railway creará automáticamente `REDIS_URL`

### 4. **Configurar Variables**
- Ir a app service → "Variables"
- Agregar: `OPENAI_API_KEY=tu_api_key`
- Las demás se crean automáticamente

### 5. **Deploy**
- Push código a GitHub
- Railway construirá y desplegará automáticamente

## 🔧 **Implementación Técnica**

### **get_checkpointer():**
```python
def get_checkpointer():
    redis_url = os.getenv("REDIS_URL")
    
    if redis_url and REDIS_CHECKPOINTER_AVAILABLE:
        # USA REDIS CHECKPOINTER REAL
        return RedisCheckpointer.from_conn_string(redis_url)
    else:
        # Fallback a memoria
        return MemorySaver()
```

### **get_store():**
```python
def get_store():
    postgres_url = os.getenv("DATABASE_URL")
    
    if postgres_url:
        # USA POSTGRESQL STORE REAL
        return PostgresStore.from_conn_string(postgres_url)
    else:
        # Fallback a memoria
        return InMemoryStore()
```

## 📦 **Dependencias Clave**
```
redis==4.5.4                          # Cliente Redis
langgraph-checkpoint-redis==2.0.2     # Redis checkpointer real
langgraph-store-postgres==0.0.9       # PostgreSQL store
psycopg2-binary==2.9.9                # Adaptador PostgreSQL
```

## ✅ **Verificación**

### **Logs Exitosos:**
```
🔴 Using Redis checkpointer for conversation state
🐘 Using PostgreSQL store for persistent data storage
```

### **Logs de Fallback:**
```
⚠️ Redis checkpointer failed: [error]
💾 Falling back to in-memory checkpointer
⚠️ PostgreSQL store failed: [error]  
💾 Falling back to in-memory store
```

## 🧪 **Testing Local**
```bash
python test_memory.py
```

Verificará:
- ✅ Conexión PostgreSQL
- ✅ Conexión Redis
- ✅ Redis checkpointer disponible
- ✅ OpenAI API key

## 🎉 **Resultado Final**

Ahora tienes memoria **100% persistente**:
- **Conversaciones** sobreviven reinicios (Redis)
- **Datos de usuario** sobreviven reinicios (PostgreSQL)
- **Fallbacks automáticos** para desarrollo local

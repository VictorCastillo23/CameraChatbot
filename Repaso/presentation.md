# LangChain

## Introducción

**¿Qué es LangChain?**
LangChain es una biblioteca diseñada para simplificar la creación de aplicaciones que utilizan modelos de lenguaje de gran tamaño(LLMs), ofreciendo herramientas para cadenas de tareas (Chains), agentes, chatbots con memoria, etc.

**Casos de uso destacados:**
* Asistentes virtuales personalizados.
* Recuperación de información en grandes volúmenes de datos.
* Automatización de tareas complejas con interacciones dinámicas.

## Instalación

**Requisitos previos:**
* Python 3.8 o superior.
* API Keys para el modelo a usar (OpenAI, Anthropic, Mistral, etc.).
* Instalación de Ollama para usar modelos locales.

**Dependencias**
```
pip install langchain
pip install langchain-community
pip install langchain-openai
pip install langchain-ollama
pip install langchain-chroma
pip install langgraph
pip install pypdf 
```

## Ecosistema

El poder de LangChain esta en combinar estas herramientas
* **LLMs:** Modelos de Lenguaje
* **Chains:** Flujo de tareas encadenadas
* **Agentes:** Toma de decisiones dinámicas
* **Memoria:** Historial persistente de interacciones

## Modelos en la Nube vs Ollama

**¿Qué es Ollama?**
Ollama permite utilizar LLMs de forma local o en entornos controlados.

**Modelos en la nube**
* Son más potentes y compatibles con nuevas funcionalidades que ofrece Langchain.
* Tienen un costo por uso. De acuerdo al número de tokens que se usa en las peticiones al modelo.

**Modelos Ollama**
* De acuerdo al número de parametros el modelo puede ser más o menos potente.
* No tiene un costo por uso, es más bien los recursos de hardware necesarios para montar el modelo.

## Cadenas - `Chains`

**¿Qué es una Chain?**
Una Chain es una secuencia de pasos o tareas que conectan entradas y salidas de modelos de lenguaje u otras herramientas. Permiten organizar flujos de trabajos complejos de manera modular.

**Tipos de Chains comunes:**
1. **LLMChain:** Encadena un modelo LLM con una entrada y salida simple.
2. **SequentialChain:** Permite encadenar múltiples LLMs o pasos en secuencia.
3. **ConversationalRetrievalchain:** Combina memoria y recuperación de información.

## Modelos con salidas estructuradas - `structured_output`

**¿Qué es una salida estructurada?**
Una salida estructurada es una respuesta en un formato específico, que permite integrar los resultados con otras herramientas o sistemas.

**Por qué es útil?**
* Simplifica la extracción de datos importantes.
* Ideal cuando se necesitan generar datos con la estructura que solicita una API.

## Modelo ChatBot - `Memory`

**Definición:**
La memoria en Langchain permite que las aplicaciones recuerden el contexto de interacciones previas.

**Aspectos clave para manejar un historial de mensajes**
1. **Store:** Base de datos para almacenar el historial.
2. **Función de recuperación:** Función para recuperar el historial de una sesión (conversación).
3. **Lista de mensajes:** Es necesario que todas las interacciones con el chatbot sean una lista de mensajes del tipo (`SystemMessage`, `HumanMessage` o `AIMessage`).

## Agentes y `Tools`

**¿Qué son los agentes y las `tools`?**
* **Agentes:** Componentes que toman decisiones sobre que acción realizar a continuación, ya sea invocar un modelo de lenguaje, usar una herramienta o realizar múltiples pasos iterativos.
* **`Tools`:** Herramientas externas que los agentes pueden invocar, como APIs, consultas a bases de datos, etc.

**Formas de crear un agente capaz de invocar `Tools`**
1. **Graph sencillo:** LangGraph permite definir flujos de trabajo mediante grafos, lo que da un mayor control en la organización.
2. **Agente ReAct:** Grafo de LangGraph preconstruido para decidir que acción tomar en función de la entrada de usuario.
3. **Agente de OpenAI:**: Agente preconstruido de LangChain especializado en usar `Tools` mediante modelos OpenAI.

### Notas:
`create_openai_tools_agent`: Es un agente que crea un plan y determina las acciones a realizar en cada paso del ciclo de ejecución.
`MessagesPlaceholder`: Sirve como un marcador de posición dentro de una plantilla de prompt que asume que una variable es una lista de mensajes.
import wikipedia
wikipedia.set_lang("en")
wikipedia.set_user_agent("my-langchain-app/1.0")

import streamlit as st
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage

# -------------------- Wikipedia Tool --------------------
api_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wiki)

def safe_wiki_run(query):
    try:
        return wiki_tool.run(query)
    except Exception:
        return "Wikipedia is unavailable. Use web search instead."

wiki = Tool(
    name="Wikipedia",
    func=safe_wiki_run,
    description="Search Wikipedia for factual, detailed information."
)

# -------------------- DuckDuckGo Tool --------------------
ducksearch = DuckDuckGoSearchRun()

# -------------------- Streamlit UI --------------------
st.title("🔎 Search Engine")
st.sidebar.title("Settings")

api_key = st.sidebar.text_input("Groq API Key", type="password")

if not api_key:
    st.warning("Please enter your Groq API key")
    st.stop()

# -------------------- Chat History Formatter --------------------
def format_chat_history(messages):
    formatted = []
    for msg in messages:
        if msg["role"] == "user":
            formatted.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted.append(AIMessage(content=msg["content"]))
    return formatted

# -------------------- Session State --------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I'm your search assistant. I can search Wikipedia and the web."
        }
    ]

# Display chat
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------- User Input --------------------
if user_prompt := st.chat_input("Ask me anything!"):
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    # LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        streaming=False
    )

    tools = [wiki, ducksearch]
    agent_kwargs = {
    "system_message": (
        "You are a smart assistant. "
        "Never call the same tool repeatedly if it fails. "
        "Always switch tools after a failure."
    )
    }

    # -------------------- Agent --------------------
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="generate",
        agent_kwargs=agent_kwargs
    )

    # -------------------- Response --------------------
    with st.chat_message("assistant"):
        callback_container = st.empty()
        st_cb = StreamlitCallbackHandler(
            callback_container,
            expand_new_thoughts=True
        )

        try:
            result = agent_executor.invoke(
                {"input": user_prompt},
                config={"callbacks": [st_cb]}
            )
            response = result["output"]
        except Exception as e:
            response = f"Error: {str(e)}"

        st.write(response)

        st.session_state["messages"].append(
            {"role": "assistant", "content": response}
        )
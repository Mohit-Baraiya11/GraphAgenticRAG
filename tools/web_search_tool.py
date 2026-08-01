from langchain_groq import ChatGroq
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def extract_web_query(user_query):
    result = llm.invoke(f"""Your job is to extract what needs to be searched on the web.

        Important: Ignore phrases like "in given PDFs", "in the document", "in uploaded files" — 
        these are just user context telling you WHERE they thought the answer might be.
        Focus on WHAT information is actually being requested.

        Examples:
        "what is current temperature in india in given PDFs" → "current temperature in india"
        "what are today's AI news in the document" → "today's AI news"
        "summarize the uploaded PDF" → "NONE"
        "what is the capital of France in given PDFs" → "capital of France"

        Return ONLY the search query string, or "NONE" if nothing needs web search.

        Query: {user_query}

        Web search query:""")
    
    return result.content.strip()

def web_search_node(state):
    user_query = state["messages"][-1].content
    
    # Extract only web-searchable part
    web_query = extract_web_query(user_query)
    
    if web_query == "NONE":
        print("No web search needed.")
        return {"retrieved_context": []}
    
    print(f"Web searching for: {web_query}")
    
    results = tavily.search(
        query=web_query,  
        max_results=5,
        search_depth="advanced",
        include_answer=True
    )
    
    contexts = []
    if results.get("answer"):
        contexts.append(f"[WEB] Direct Answer: {results['answer']}")
    
    for result in results["results"]:
        context = f"[WEB] Source: {result['url']}\n{result['content']}"
        contexts.append(context)

    print(f"Web search found {len(contexts)} results.")
    return {
        "retrieved_context": contexts,
        "current_context":context,
    }
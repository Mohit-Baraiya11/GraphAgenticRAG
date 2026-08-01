import nest_asyncio
nest_asyncio.apply()  # must be at top before anything else
import asyncio
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)


def extract_github_url(text: str):
    pattern = r'https?://github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)'
    match = re.search(pattern, text)
    if match:
        return match.group(1), match.group(2)
    return None, None


async def fetch_github_context(owner: str, repo: str, query: str) -> list[str]:
    client = MultiServerMCPClient({
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")},
            "transport": "stdio"
        }
    })

    tools = await client.get_tools()

    def get_tool(name):
        return next((t for t in tools if t.name == name), None)

    contexts = []

    # Always fetch README
    readme_tool = get_tool("get_file_contents")
    if readme_tool:
        try:
            result = await readme_tool.ainvoke({
                "owner": owner,
                "repo": repo,
                "path": "README.md"
            })
            contexts.append(f"[GITHUB README]\n{str(result)[:3000]}")
        except Exception as e:
            print(f"README fetch failed: {e}")

    # Get repo info
    repo_tool = get_tool("get_repository")
    if repo_tool:
        try:
            result = await repo_tool.ainvoke({
                "owner": owner,
                "repo": repo
            })
            contexts.append(f"[GITHUB REPO INFO]\n{str(result)[:2000]}")
        except Exception as e:
            print(f"Repo info fetch failed: {e}")

    # Search code for specific queries
    query_lower = query.lower()
    if any(word in query_lower for word in ["bug", "code", "function", "error", "fix", "review", "file"]):
        search_tool = get_tool("search_code")
        if search_tool:
            try:
                result = await search_tool.ainvoke({
                    "q": f"repo:{owner}/{repo} {query[:50]}"
                })
                contexts.append(f"[GITHUB CODE SEARCH]\n{str(result)[:2000]}")
            except Exception as e:
                print(f"Code search failed: {e}")

    return contexts


def github_rag_node(state):
    user_query = state["messages"][-1].content
    owner, repo = extract_github_url(user_query)
    
    if not owner or not repo:
        return {
            "retrieved_context": ["No valid GitHub repository URL found."],
            "current_context": ["No valid GitHub repository URL found."]
        }
    
    
    # Run async MCP client in sync context
    contexts = asyncio.run(fetch_github_context(owner, repo, user_query))
    
    
    return {
        "retrieved_context": contexts,
        "current_context": contexts
    }
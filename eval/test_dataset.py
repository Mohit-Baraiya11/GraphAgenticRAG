from tavily import hybrid_rag


router_test_cases = [
    {
        "query": "What are Mohit's skills in Generative AI?",
        "expected": ["hybrid_rag"],
        "reason": "Directly in resume skills section"
    },
    {
        "query": "Where did Mohit do his internship?",
        "expected": ["hybrid_rag"],
        "reason": "Experience section in resume"
    },
    {
        "query": "What projects has Mohit built?",
        "expected": ["hybrid_rag"],
        "reason": "Projects section in resume"
    },
    {
        "query": "What is Mohit's educational background?",
        "expected": ["hybrid_rag"],
        "reason": "Education section in resume"
    },
    {
        "query": "What frameworks does the candidate know?",
        "expected": ["hybrid_rag"],
        "reason": "Skills section"
    },
    {
        "query": "Tell me about SourceMind AI project",
        "expected": ["hybrid_rag"],
        "reason": "Project details in resume"
    },
    {
        "query": "What databases does Mohit work with?",
        "expected": ["hybrid_rag"],
        "reason": "Skills section"
    },
    {
        "query": "Summarize the attached resume",
        "expected": ["hybrid_rag"],
        "reason": "Document summarization"
    },
    {
        "query": "What is the candidate's SGPA?",
        "expected": ["hybrid_rag"],
        "reason": "Education section"
    },
    {
        "query": "What did Mohit build at Fuerte Developers?",
        "expected": ["hybrid_rag"],
        "reason": "Experience section"
    },

    # web_search cases — requires internet
    {
        "query": "What is the current price of Bitcoin?",
        "expected": ["web_search"],
        "reason": "Real-time external data"
    },
    {
        "query": "What are the latest AI trends in 2025?",
        "expected": ["web_search"],
        "reason": "Current external information"
    },
    {
        "query": "What is the weather in Rajkot today?",
        "expected": ["web_search"],
        "reason": "Real-time weather data"
    },
    {
        "query": "Who won the IPL 2025?",
        "expected": ["web_search"],
        "reason": "Current events"
    },
    {
        "query": "What is the latest version of LangChain?",
        "expected": ["web_search"],
        "reason": "Current software version"
    },

    # github_rag cases
    {
        "query": "Explain this repo: https://github.com/Mohit-Baraiya11/Blueprint-agent",
        "expected": ["github_rag"],
        "reason": "GitHub URL present"
    },
    {
        "query": "Find bugs in https://github.com/Mohit-Baraiya11/Blueprint-agent",
        "expected": ["github_rag"],
        "reason": "GitHub URL with code review request"
    },
    {
        "query": "What does https://github.com/Mohit-Baraiya11/SourceMind-AI do?",
        "expected": ["github_rag"],
        "reason": "GitHub URL present"
    },

    # direct cases — casual conversation
    {
        "query": "Hi",
        "expected": ["direct"],
        "reason": "Pure greeting"
    },
    {
        "query": "Thanks for your help",
        "expected": ["direct"],
        "reason": "Casual acknowledgment"
    },
    {
        "query": "What can you do?",
        "expected": ["direct"],
        "reason": "Capability question, no document/web needed"
    },

    # graph_rag cases
    {
        "query": "What technologies are connected to the SourceMind AI project?",
        "expected": ["graph_rag"],
        "reason": "Entity relationship query"
    },
    {
        "query": "How are Mohit's projects related to each other?",
        "expected": ["graph_rag"],
        "reason": "Relationship between entities"
    },

    # edge cases — tricky ones
    {
        "query": "What is LangGraph?",
        "expected": ["web_search"],
        "reason": "General knowledge question not about Mohit specifically"
    },
    {
        "query": "Is Mohit good at Python?",
        "expected": ["hybrid_rag"],
        "reason": "About candidate, answerable from resume"
    },
    {
        "query": "What is Mohit's email?",
        "expected": ["hybrid_rag"],
        "reason": "Contact info in resume header"
    },
]


# golden dataset
hybrid_rag_test_cases = [
    {
        "question": "What are Mohit's Generative AI skills?",
        "ground_truth": "Mohit's Generative AI skills include RAG, Hybrid RAG, Graph RAG, Agentic AI with LangGraph, HITL, tool calling, Pinecone, HuggingFace Embeddings, SemanticChunker, and Groq API with LLaMA 3."
    },
    {
        "question": "Where did Mohit do his internship and what did he build?",
        "ground_truth": "Mohit did his internship at Fuerte Developers from January 2025 to June 2025 in Rajkot, India. He built Aapka Vyapar, a cross-platform business management app for small shopkeepers using Flutter and Dart, with Firebase authentication and Firestore."
    },
    {
        "question": "What is Mohit's educational background?",
        "ground_truth": "Mohit is pursuing B.E. in AI and Data Science at Government Engineering College Rajkot from 2025. He completed Diploma in Computer Science and Engineering from R.K. University with SGPA of 8.90."
    },
    {
        "question": "What is the SourceMind AI project?",
        "ground_truth": "SourceMind AI is a Hybrid RAG Chatbot built by Mohit using Python, FastAPI, LangChain, Pinecone, HuggingFace, Groq LLaMA 3, and Supabase. It combines dense HuggingFace embeddings with BM25 sparse vectors on Pinecone using SemanticChunker parent-child chunking for PDF and YouTube transcript Q&A."
    },
    {
        "question": "What programming languages does Mohit know?",
        "ground_truth": "Mohit knows Python and SQL."
    },
    {
        "question": "What is the Blueprint Agent project?",
        "ground_truth": "Blueprint Agent is a LangGraph Agentic Pipeline that turns a vague project idea into a downloadable PDF blueprint covering market validation, system architecture, tech stack, and phase-wise build plan. It uses an 8-node LangGraph pipeline with Human-in-the-Loop via interrupt(), Tavily web search for validation, and generates skill-aware stack recommendations."
    },
    {
        "question": "What databases does Mohit work with?",
        "ground_truth": "Mohit works with Supabase (PostgreSQL, Auth) and Pinecone."
    },
    {
        "question": "What is Mohit's contact email?",
        "ground_truth": "Mohit's contact email is mohitbaraiya761@gmail.com."
    },
    {
        "question": "What ML algorithms does Mohit know?",
        "ground_truth": "Mohit knows Data Preprocessing, EDA, Linear Regression, Logistic Regression, Decision Tree, Random Forest, K-Means, and Ensemble Learning."
    },
    {
        "question": "What deep learning architectures does Mohit know?",
        "ground_truth": "Mohit knows ANN, CNN, RNN, LSTM, GRU, and Transformer Architecture using Keras and TensorFlow."
    },
]
FROM python:3.11-slim

# Install Node.js for GitHub MCP server
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install npx globally
RUN npm install -g @modelcontextprotocol/server-github

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv
RUN pip install uv

# Install dependencies
RUN uv sync --frozen --no-dev && rm -rf /root/.cache

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
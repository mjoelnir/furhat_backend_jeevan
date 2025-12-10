# Conversational Agent Project

This project implements an advanced conversational agent that can:

- Ingest documents (e.g., PDFs) and store them in a Chroma vector database for **retrieval-augmented generation (RAG)**.
- Dynamically route queries to web search or vectorstore retrieval.
- Integrate with external APIs (Foursquare, OpenStreetMap, Overpass, IPData) for location-based or IP-based queries.
- Generate natural language responses via Large Language Models (LLMs) such as HuggingFace Transformers or Llama Cpp.
- Leverage GPU acceleration for improved performance on EC2 instances.

It includes **FastAPI** endpoints for synchronous/asynchronous interaction, **LangGraph**-based state machines for multi-step conversation flows, and classification tools.

---

## Table of Contents

- [Conversational Agent Project](#conversational-agent-project)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Requirements & Installation](#requirements--installation)
  - [Configuration](#configuration)
    - [Creating Your .env File](#creating-your-env-file)
    - [Using direnv for Environment Management](#using-direnv-for-environment-management)
  - [EC2 Instance Setup](#ec2-instance-setup)
    - [Hardware Specifications](#hardware-specifications)
    - [Storage Setup](#storage-setup)
    - [Model Management](#model-management)
    - [AWS Configuration and Server Setup](#aws-configuration-and-server-setup)
  - [Connecting to the EC2 Instance](#connecting-to-the-ec2-instance)
    - [Using Terminal SSH](#using-terminal-ssh)
    - [Using VS Code Remote - SSH](#using-vs-code-remote---ssh)
  - [Cloning the Repository & Running the Dev Container on the EC2 Instance](#cloning-the-repository--running-the-dev-container-on-the-ec2-instance)
  - [Starting the Dev Container](#starting-the-dev-container)
    - [Locally](#locally)
    - [On the EC2 Instance via VS Code](#on-the-ec2-instance-via-vs-code)
    - [Manual Docker Run on the EC2 Instance](#manual-docker-run-on-the-ec2-instance)
  - [Usage](#usage)
    - [Running the FastAPI Server](#running-the-fastapi-server)
    - [Interactive Conversations & Tests](#interactive-conversations--tests)
    - [Interacting with the API Endpoints](#interacting-with-the-api-endpoints)
      - [1. Interactive API Documentation](#1-interactive-api-documentation)
      - [2. Using cURL Commands](#2-using-curl-commands)
      - [3. Using a Python Script with Requests](#3-using-a-python-script-with-requests)
  - [Managing Caches](#managing-caches)
    - [Clearing All Caches](#clearing-all-caches)
      - [1. Using the API Endpoint](#1-using-the-api-endpoint)
      - [2. Using Python Script](#2-using-python-script)
      - [3. Manual Cache Clearing](#3-manual-cache-clearing)
  - [Managing Dependencies with Poetry and pip](#managing-dependencies-with-poetry-and-pip)
    - [Activating the Poetry Environment](#activating-the-poetry-environment)
    - [Additional Troubleshooting Steps](#additional-troubleshooting-steps)
  - [Key Components](#key-components)
  - [Notes](#notes)
  - [Contributing](#contributing)

---

## Project Structure

A simplified view of the repository (showing the most important directories and files):

```
.
├── furhat_skills/               # (Optional) Skills or scripts for Furhat robot integration
├── middleware/
│   └── main.py                  # FastAPI endpoints for document agent interaction
├── my_furhat_backend/
│   ├── agents/
│   │   ├── document_agent.py            # State-graph-based DocumentAgent for RAG conversations
│   │   ├── test_2_conversational_agent.py  # Extended test workflow with dynamic routing & grading
│   │   └── test_conversational_agent.py    # Basic RAG-based conversation test
│   ├── api_clients/
│   │   ├── foursquare_client.py    # Foursquare API client
│   │   ├── osm_client.py           # OSM (Nominatim) API client
│   │   ├── overpass_client.py      # Overpass API client
│   │   └── ipdata_client.py        # IPData API client
│   ├── config/
│   │   └── settings.py             # Loads environment variables (dotenv)
│   ├── ingestion/
│   │   └── CMRPublished.pdf        # Example PDF for ingestion (or other docs)
│   ├── llm_tools/
│   │   └── tools.py                # Defines location-based search tools referencing api_clients
│   ├── models/
│   │   ├── chatbot_factory.py      # Factory to build HuggingFace/Llama-based chatbots
│   │   ├── classifier.py           # Zero-shot classifier
│   │   ├── llm_factory.py          # Creates HuggingFace/Llama LLMs
│   │   └── model_pipeline.py       # Manages HF pipelines (auth, loading, etc.)
│   ├── RAG/
│   │   └── rag_flow.py             # RAG logic (loading, chunking, storing in Chroma)
│   ├── utils/
│   │   └── util.py                 # Utility functions for text cleaning & formatting
│   └── main.py                     # FastAPI app with /ask, /transcribe, /response endpoints
├── tests/                          # Additional tests or integration tests
├── .env                            # Environment variables (API keys, etc.) – typically in .gitignore
├── .envrc                          # direnv configuration file
├── pyproject.toml                  # Poetry or other build-system configuration
├── poetry.lock                     # Poetry lockfile
└── README.md                       # Project documentation (this file)
```

---

## EC2 Instance Setup

### Hardware Specifications

The project is optimized for running on an EC2 instance with the following specifications:

- Instance Type: g4dn.xlarge (or similar GPU-enabled instance)
- GPU: NVIDIA Tesla T4
- Memory: 16 GB
- Storage: 100 GB GP3

### Storage Setup

The EC2 instance uses the following storage structure:

```
/mnt/
├── data/
│   ├── documents/          # PDF documents for RAG
│   └── vector_store/      # Chroma vector store
├── models/
│   ├── caches/           # Model caches
│   │   ├── huggingface/  # HuggingFace model cache
│   │   └── document_agent/ # Document agent cache
│   └── gguf/             # GGUF model files
```

### Model Management

1. **Model Files:**

   - GGUF models are stored in `/mnt/models/gguf/`
   - Supported models include:
     - Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
     - Mistral-Nemo-Instruct-2407-Q4_K_M.gguf
     - Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf
     - SmolLM2-1.7B-Instruct-Q4_K_M.gguf

2. **Cache Management:**

   - Model caches are stored in `/mnt/models/caches/`
   - To clear caches:
     ```bash
     sudo rm -rf /mnt/models/caches/huggingface/* /mnt/models/caches/document_agent/*
     ```

3. **Vector Store:**
   - Located at `/mnt/data/vector_store/`
   - To clear vector store:
     ```bash
     sudo rm -rf /mnt/data/vector_store/*
     ```

### GPU Optimization

The project includes several GPU optimizations:

1. **LLM Processing:**

   - Uses CUDA acceleration for model inference
   - Optimized batch processing for embeddings
   - GPU memory management with automatic cache clearing

2. **RAG System:**

   - GPU-accelerated document embeddings
   - Optimized chunking parameters for better performance
   - Parallel processing for document chunking
   - Efficient vector store operations

3. **Memory Management:**
   - Automatic GPU cache clearing
   - Memory usage monitoring
   - Efficient batch processing

### AWS Configuration and Server Setup

1. **Security Groups**
   - Inbound Rules:
     - SSH (Port 22) from your IP
     - HTTP (Port 80) from anywhere
     - HTTPS (Port 443) from anywhere
     - Custom TCP (Port 8000) for FastAPI server
   - Outbound Rules:
     - All traffic allowed

2. **IAM Roles and Permissions**
   - Create an IAM role with the following permissions:
     - AmazonS3FullAccess (for model storage)
     - CloudWatchLogsFullAccess (for logging)
     - AmazonEC2ContainerRegistryFullAccess (if using ECR)

3. **Server Configuration**
   - Install required system packages:
     ```bash
     sudo apt-get update
     sudo apt-get install -y python3-pip python3-venv nvidia-driver-525
     ```
   - Configure NVIDIA drivers:
     ```bash
     sudo apt-get install -y nvidia-cuda-toolkit
     nvidia-smi  # Verify GPU is recognized
     ```
   - Set up environment variables:
     ```bash
     echo 'export PYTHONPATH=/mnt/data/app/my_furhat_backend:$PYTHONPATH' >> ~/.bashrc
     echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
     source ~/.bashrc
     ```

4. **Production Deployment**
   - Use systemd service for automatic startup:
     ```bash
     sudo nano /etc/systemd/system/furhat-backend.service
     ```
   - Add the following configuration:
     ```ini
     [Unit]
     Description=Furhat Backend Service
     After=network.target

     [Service]
     User=ubuntu
     WorkingDirectory=/mnt/data/app/my_furhat_backend
     Environment="PATH=/mnt/data/app/my_furhat_backend/.venv/bin"
     ExecStart=/mnt/data/app/my_furhat_backend/.venv/bin/uvicorn middleware.main:app --host 0.0.0.0 --port 8000
     Restart=always

     [Install]
     WantedBy=multi-user.target
     ```
   - Enable and start the service:
     ```bash
     sudo systemctl enable furhat-backend
     sudo systemctl start furhat-backend
     ```

5. **Monitoring and Logging**
   - Set up CloudWatch logging:
     ```bash
     sudo apt-get install -y amazon-cloudwatch-agent
     ```
   - Configure log rotation:
     ```bash
     sudo nano /etc/logrotate.d/furhat-backend
     ```
   - Add configuration:
     ```
     /mnt/data/app/my_furhat_backend/logs/*.log {
         daily
         rotate 7
         compress
         delaycompress
         missingok
         notifempty
         create 0640 ubuntu ubuntu
     }
     ```

---

## Recent Additions and Defaults (skill + backend)

- **Perception websocket**: Robot streams frames to `/ws/perception` (binary prefix `0x01` video JPEG, `0x02` audio). Text messages carry `hello`, `turn`, `name_update`. Server soft-fails if face/voice libs are missing; user IDs are created on-demand for stats.
- **Trivia flow**: Kotlin skill calls `/quiz/question`, `/trivia/turn` (LLM phrasing/feedback), and `/memory/trivia` (stats). Local cache + backend persistence (`trivia_stats` table).
- **Language handling**: Heuristic EN/NO detector client-side; explicit language pinning via `LanguageManager` with Polly voices (`Kendra-Neural` EN, `Ida-Neural` NO). Backend language hinting in prompts; placeholder server-side lang detect.
- **URLs/IPs**: Default `BACKEND_URL` in the skill points to laptop IP (override via `BACKEND_URL` env). Robot IP tracked in params. Perception WS URL derives from `BACKEND_URL` (`ws://<host>:8000/ws/perception`).
- **RAG**: Lightweight BM25 (no vector DB) over `DOCUMENTS_PATH` with preference for `qa_pairs.json` when present; falls back to PDFs/txt. Chunk size 800 / overlap 150.
- **LLM defaults**: Backend uses Ollama (`llama3.2:latest`) by default; HF/LlamaCpp paths are guarded behind torch/transformers availability. System prompt in `config/settings.py` defines the Kaia persona and strict language policy.
- **Storage/layout**: Backend defaults to local `.cache` for models/caches/docs/vector store/DB (SQLite `furhat_memory.db`). Tables: `users`, `conversations`, `turns`, `trivia_stats`.
- **Ingestion helper**: `ingestion/web_ingest.py` fetches PDFs via DuckDuckGo HTML + regex (best-effort), storing into a folder you can set as `DOCUMENTS_PATH`.


## Requirements & Installation

This project uses a hybrid dependency management approach: some dependencies are installed via pip into your environment, and others are managed by Poetry (tracked in the `poetry.lock` file).

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Install Dependencies via Poetry**  
   If you haven't already, install Poetry globally. Then run:

   ```bash
   poetry lock
   poetry install
   ```

   This will update the `poetry.lock` file and install dependencies into Poetry's virtual environment.

3. **Activate the Poetry Environment**  
   It's important to activate your Poetry environment before installing any additional pip dependencies. See the [Activating the Poetry Environment](#activating-the-poetry-environment) section below.

4. **(Optional) Install Additional pip Dependencies**  
   If there is a `requirements.txt` file with extra dependencies, activate your Poetry environment first and then run:

   ```bash
   pip install -r requirements.txt
   ```

   If you are having issues with building the wheel for `llama.cpp` remove it from the requirements file and download the applicable version manually based on your hardware. More details can be found here: **https://github.com/abetlen/llama-cpp-python**.

---

## Configuration

### Creating Your .env File

Because the `.env` file is typically listed in `.gitignore`, you'll need to create your own locally:

1. **Create a `.env` file** in the root directory of the project (same level as `pyproject.toml`).
2. **Add your API keys and environment variables.** For example:
   ```bash
   FSQ_KEY=<YOUR_FOURSQUARE_API_KEY>
   IP_KEY=<YOUR_IPDATA_API_KEY>
   HF_KEY=<YOUR_HUGGINGFACE_API_KEY>
   ```
3. **Save the file.** This file will be loaded at runtime by `my_furhat_backend/config/settings.py`.

### Using direnv for Environment Management

If you use **direnv** to automatically load environment variables:

- You might see an error like:
  ```
  direnv: error /Users/{home_directory}/my_furhat_backend/.envrc is blocked. Run `direnv allow` to approve its content
  ```
- To approve the content of `.envrc`, run:

  ```bash
  direnv allow
  ```

  You may also need to run:

  ```bash
  source ~/.zshrc
  ```

  This command will load the file and export the necessary environment variables.

- Your `.envrc` may include:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  ```
  This ensures that your local binaries are in your PATH.

---

## Connecting to the EC2 Instance

This section provides instructions for connecting to the EC2 instance hosting the Conversational Agent both via a terminal and through VS Code.

### Using Terminal SSH

1. **Instance Details:**

   - **Instance ID:** `i-04ea4401b15d43473`
   - **Public DNS:** `ec2-51-21-200-54.eu-north-1.compute.amazonaws.com`
   - **Private Key File:** `Furhat_W25.pem` (the key used to launch the instance, you will have to create your own)

2. **Prepare Your Private Key:**
   Ensure your private key is secure (i.e., not publicly viewable) by running:

   ```bash
   chmod 400 "path/to/Furhat_W25.pem"
   ```

3. **Connect Using SSH:**
   Open your terminal and run:
   ```bash
   ssh -i "path/to/Furhat_W25.pem" ec2-user@ec2-51-21-200-54.eu-north-1.compute.amazonaws.com
   ```
   When prompted to verify the host's authenticity, type **yes**. Once connected, you'll be logged in as `ec2-user` on your EC2 instance.

### Using VS Code Remote - SSH

VS Code's Remote - SSH extension allows you to seamlessly edit and develop on your remote EC2 instance.

1. **Install the Remote - SSH Extension:**

   - Open VS Code and go to the Extensions view (press `Cmd+Shift+X` on Mac).
   - Search for "Remote - SSH" (by Microsoft) and install it.

2. **Configure Your SSH Host in VS Code:**

   - Open the Command Palette (`Cmd+Shift+P`).
   - Select **Remote-SSH: Add New SSH Host...**
   - Enter the following SSH command:
     ```bash
     ssh -i "path/to/Furhat_W25.pem" ec2-user@ec2-51-21-200-54.eu-north-1.compute.amazonaws.com
     ```
   - Choose to save the configuration (this will typically update your `~/.ssh/config` file).

3. **Connect to the EC2 Instance:**
   - Open the Command Palette again and select **Remote-SSH: Connect to Host...**
   - Choose your newly added host.
   - Accept any host key verification prompts.
   - VS Code will then open a new window connected to your EC2 instance, allowing you to work on files and run terminals on the remote server.

---

#### Cloning the Repository & Setting Up Docker on the EC2 Instance

If you haven't already set up your EC2 instance with Git and Docker, follow these steps:

1. **Install Git:**

   ```bash
   sudo yum update -y
   sudo yum install git -y
   ```

2. **Install Docker:**

   ```bash
   sudo yum install docker -y
   ```

   _(For Amazon Linux 2, you might also use `sudo amazon-linux-extras install docker -y` if available.)_

3. **Start Docker and Configure Permissions:**

   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker ec2-user
   newgrp docker  # Alternatively, log out and log back in to apply the group change
   ```

4. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

5. **Build and Run the Dev Container:**

   Assuming your project structure is as follows:

   ```
   /my-furhat-backend
   ├── .devcontainer
   │   ├── devcontainer.json
   │   └── Dockerfile
   ├── pyproject.toml
   ├── poetry.lock
   └── ... (other project files)
   ```

   From the project root, build the Docker image:

   ```bash
   docker build -t my-furhat-backend -f .devcontainer/Dockerfile .
   ```

   Then, run the container interactively:

   ```bash
   docker run -it --rm -p 8000:8000 -v "$(pwd)":/app my-furhat-backend
   ```

   This command:

   - Runs the container in interactive mode with a terminal (`-it`)
   - Automatically removes the container when it exits (`--rm`)
   - Maps port 8000 of the container to port 8000 on the host (`-p 8000:8000`)
   - Mounts your current project directory into `/app` inside the container (`-v "$(pwd)":/app`)

---

## Starting the Dev Container

This section explains how to launch an interactive development container using VS Code's Remote - Containers extension. You can use this container for interactive development either locally or on your EC2 instance.

### Locally

1. **Open in Dev Container:**
   - Open your project in VS Code.
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) and select **Remote-Containers: Reopen in Container**.
   - VS Code will build the container based on your `.devcontainer` configuration and open a new window connected to the container.
   - You now have an interactive development environment with all your dependencies installed.

### On the EC2 Instance

1. **Clone Your Repository on the EC2 Instance:**

   - SSH into your EC2 instance:
     ```bash
     ssh -i "path/to/Furhat_W25.pem" ec2-user@ec2-51-21-200-54.eu-north-1.compute.amazonaws.com
     ```
   - Clone your repository:
     ```bash
     git clone https://github.com/yourusername/yourproject.git
     cd yourproject
     ```

2. **Connect to the EC2 Instance Using VS Code Remote - SSH:**

   - On your local machine, open VS Code and use the Remote - SSH extension to connect to your EC2 instance (as described in the [Connecting to the EC2 Instance](#connecting-to-the-ec2-instance) section).

3. **Open the Dev Container on EC2:**
   - In the VS Code window connected to the EC2 instance, open the Command Palette and select **Remote-Containers: Reopen in Container**.
   - VS Code will use the dev container configuration (the same `.devcontainer` folder) to build and attach the dev container on the EC2 instance.
   - Now, you have an interactive development environment running on your EC2 instance that mirrors your local dev container setup.

---

## Usage

### Running the FastAPI Server

To start the server (after ensuring your environment is activated):

```bash
poetry run uvicorn middleware.main:app --host 0.0.0.0 --port 8000
```

Or, if using pip/venv:

```bash
uvicorn middleware.main:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

- `POST /ask` — Synchronously processes a user query and returns an answer.
- `POST /transcribe` — Asynchronously handles transcriptions (stores response for later retrieval).
- `GET /response` — Fetches the latest response generated by the agent.
- `POST /get_docs` - Fetches the name of the document that is the most similar to the users request

### Interactive Conversations & Tests

There are multiple ways to test or interact with the conversation agent:

1. **Basic RAG-based Conversation**

   ```bash
   poetry run python my_furhat_backend/agents/test_conversational_agent.py
   ```

   This script starts a conversation loop using document ingestion and retrieval.

2. **Extended Workflow with Grading & Routing**

   ```bash
   poetry run python my_furhat_backend/agents/test_2_conversational_agent.py
   ```

   This version adds dynamic routing (vector store vs. web search) and grading chains (document/answer grading).

3. **Middleware Service**  
   If you have a middleware service at `middleware/main.py`, run:
   ```bash
   poetry run python middleware.main.py
   ```
   Or, if it's a FastAPI service:
   ```bash
   uvicorn middleware.main:app --reload
   ```

### Interacting with the API Endpoints

The middleware provides several endpoints for interacting with the document agent:

1. **Interactive API Documentation**

   - Access the Swagger UI at `http://localhost:8000/docs`
   - Access the ReDoc UI at `http://localhost:8000/redoc`

2. **Using cURL Commands**

   a. **Ask a Question** (Synchronous):

   ```bash
   curl -X POST "http://localhost:8000/ask" \
        -H "Content-Type: application/json" \
        -d '{"content": "What is the MIMIR project?"}'
   ```

   b. **Transcribe** (Asynchronous):

   ```bash
   curl -X POST "http://localhost:8000/transcribe" \
        -H "Content-Type: application/json" \
        -d '{"content": "Tell me about the project timeline"}'
   ```

   c. **Get Response** (For async requests):

   ```bash
   curl "http://localhost:8000/response"
   ```

   d. **Get Documents**:

   ```bash
   curl -X POST "http://localhost:8000/get_docs" \
        -H "Content-Type: application/json" \
        -d '{"content": "Show me the annual report"}'
   ```

   e. **Engage** (Generate follow-up):

   ```bash
   curl -X POST "http://localhost:8000/engage" \
        -H "Content-Type: application/json" \
        -d '{
             "document": "NorwAi annual report 2023.pdf",
             "answer": "The project aims to investigate copyright-protected content in language models."
           }'
   ```

3. **Using a Python Script with Requests**

   ```python
   import requests
   import json

   BASE_URL = "http://localhost:8000"

   def ask_question(question: str) -> dict:
       response = requests.post(
           f"{BASE_URL}/ask",
           json={"content": question}
       )
       return response.json()

   def transcribe(text: str) -> dict:
       response = requests.post(
           f"{BASE_URL}/transcribe",
           json={"content": text}
       )
       return response.json()

   def get_response() -> dict:
       response = requests.get(f"{BASE_URL}/response")
       return response.json()

   def get_docs(query: str) -> dict:
       response = requests.post(
           f"{BASE_URL}/get_docs",
           json={"content": query}
       )
       return response.json()

   def engage(document: str, answer: str) -> dict:
       response = requests.post(
           f"{BASE_URL}/engage",
           json={"document": document, "answer": answer}
       )
       return response.json()

   # Example usage
   if __name__ == "__main__":
       # Ask a question
       result = ask_question("What is the MIMIR project?")
       print("Answer:", result["response"])

       # Generate a follow-up
       follow_up = engage(
           "NorwAi annual report 2023.pdf",
           result["response"]
       )
       print("Follow-up:", follow_up["prompt"])
   ```

### Key Components

1. **Document Agent**

   - Handles document ingestion and RAG-based conversations
   - Uses GPU-accelerated embeddings for efficient retrieval
   - Implements state-graph-based conversation flow
   - Generates natural follow-up questions

2. **RAG System**

   - Efficient document chunking with optimized parameters
   - GPU-accelerated embeddings using HuggingFace or LlamaCpp
   - Chroma vector store for fast similarity search
   - Cross-encoder reranking for improved relevance

3. **LLM Integration**

   - Support for multiple GGUF models
   - GPU-accelerated inference
   - Optimized memory management
   - Efficient batch processing

4. **API Layer**
   - FastAPI-based REST endpoints
   - Asynchronous request handling
   - Real-time response generation
   - Document management and retrieval

### Notes

1. **GPU Usage**

   - Monitor GPU memory usage with `nvidia-smi`
   - Clear GPU cache when needed using the provided utilities
   - Adjust batch sizes based on available GPU memory

2. **Performance Optimization**

   - Use appropriate chunk sizes for document processing
   - Monitor vector store size and performance
   - Clear caches periodically to prevent memory issues
   - Use batch processing for large document sets

3. **Error Handling**

   - Check GPU memory before large operations
   - Monitor vector store integrity
   - Handle model loading errors gracefully
   - Implement proper error logging

4. **Security**
   - Secure API endpoints with authentication
   - Protect sensitive document content
   - Implement rate limiting
   - Monitor system resources

---

## Managing Caches

The system uses several types of caches to improve performance:

- Question cache: Stores previously asked questions and answers
- Context cache: Stores document context for faster retrieval
- GPU cache: Stores model weights and computations
- Conversation memory: Stores recent conversation history
- Summary cache: Stores document summaries

### Clearing All Caches

There are several ways to clear the caches:

#### 1. Using the API Endpoint

You can clear all caches by making a POST request to the `/clear_caches` endpoint:

```bash
curl -X POST http://localhost:8000/clear_caches
```

#### 2. Using Python Script

You can clear caches programmatically using the `DocumentAgent` class:

```python
from my_furhat_backend.agents.document_agent import DocumentAgent

# Initialize the agent
agent = DocumentAgent()

# Clear all caches
agent.clear_all_caches()
```

#### 3. Manual Cache Clearing

You can also manually clear the caches by:

1. Deleting the cache files:

   ```bash
   rm -rf /mnt/models/caches/huggingface/question_cache.json
   ```

2. Clearing GPU cache:

   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. Restarting the FastAPI server:

   ```bash
   # Stop the current server
   pkill -f "uvicorn middleware.main:app"

   # Start a new server
   uvicorn middleware.main:app --host 0.0.0.0 --port 8000
   ```

Note: Caches are automatically cleared when:

- The context window is exceeded
- Processing errors occur
- The server is restarted

#### 4. Command Line Cache Clearing

```bash
python -c "from my_furhat_backend.agents.document_agent import DocumentAgent; agent = DocumentAgent(); agent.clear_all_caches()"
```

---

## Managing Dependencies with Poetry and pip

This project employs a hybrid dependency strategy:

- **Poetry:** Manages the core dependencies (tracked in `poetry.lock`).
- **pip:** Some additional dependencies may be installed via pip.

**Correct Flow:**

1. Install Poetry and run:
   ```bash
   poetry lock
   poetry install
   ```
2. **Activate the Poetry environment** (see [Activating the Poetry Environment](#activating-the-poetry-environment)).
3. Once inside the Poetry environment, if needed, install extra dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Activating the Poetry Environment

When using **Poetry**, you typically run:

```bash
poetry shell
```

If that command does not work as expected or you prefer a manual approach, activate the environment with:

```bash
source $(poetry env info --path)/bin/activate
```

This command retrieves the virtual environment's path from `poetry env info --path` and manually activates it.

### Additional Troubleshooting Steps

- **Project Directory:** Ensure you're in the directory containing your `pyproject.toml` file, as some Poetry commands require it.
- **Check for Aliases:** Verify there is no shell alias or function named `poetry` or `shell` interfering with the command.
- **Reinstall/Upgrade Poetry:** Although version 2.0.1 should work, if issues persist, consider reinstalling or updating Poetry.
- **Manual Activation:** If `poetry shell` fails, using `source $(poetry env info --path)/bin/activate` should let you work within your Poetry virtual environment.

---

## Key Components

- **`main.py` in `my_furhat_backend/`**: FastAPI app exposing `/ask`, `/transcribe`, and `/response` endpoints.
- **`rag_flow.py` in `RAG/`**: Handles document ingestion, chunking, storage in a Chroma vector store, and retrieval with optional cross-encoder reranking.
- **`document_agent.py`**: Defines a **StateGraph** workflow for capturing user input, retrieving context, checking uncertainty, and generating responses.
- **`chatbot_factory.py`, `llm_factory.py`, `model_pipeline.py`**: Provide factory methods and pipelines for constructing language models (HuggingFace or Llama Cpp) and chatbots.
- **`llm_tools/tools.py`**: Integration points (tools) that the agent can invoke for location or place queries, referencing the various API clients in `api_clients/`.
- **`middleware/main.py`**: Additional FastAPI or bridging service to connect the conversation agent with other systems.
- **`tests/`**: Contains additional tests or integration checks.

---

## Notes

- Currently, the tools in `llm_tools/tools.py` and the APIs in `api_clients` are not being used. They were created at the early stages of the project when the original scope was for an agent acting as a concierge.

---

## Contributing

Contributions are welcome! If you have ideas or bug fixes:

1. Fork this repository.
2. Create a new branch for your changes.
3. Submit a pull request describing your enhancements.

---

**Enjoy using your Conversational Agent!** If you have any questions or run into issues, feel free to open an issue or reach out for support.

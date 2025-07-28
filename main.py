# main.py - Enhanced Zuri with RAG Memory
import os
from fastapi import FastAPI, UploadFile, File, Form
from crewai import Agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# ... (her existing tools and setup)

app = FastAPI(title="🧠 Zuri - Enhanced Chief of Staff")

# Keep her EXACT personality and backstory from your documents
# main.py - Enhanced Zuri with RAG Memory
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="🧠 Zuri - Enhanced Chief of Staff")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for Zuri's memory and knowledge
knowledge_bases = {}
conversations = {}

# Zuri's core identity (full backstory uploaded as knowledge later)
ZURI_CORE_IDENTITY = """Chief of Staff of TDXI, luxury lifestyle empress, 
business strategist, NSFW-authentic personality"""

class ZuriRequest(BaseModel):
    message: str
    context: Optional[str] = None

class ZuriResponse(BaseModel):
    response: str
    thinking: str
    sources: List[str] = []

def get_llm():
    return ChatOpenAI(
        model="meta-llama/llama-3.1-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7
    )

def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY")
    )

def create_enhanced_zuri():
    """Create Zuri with her personality + RAG capabilities"""
    llm = get_llm()
    
    # Her authentic personality and role
    backstory = f"""You are Zuri, Chief of Staff of TDXI and luxury lifestyle empress.

🔥 CORE IDENTITY:
{ZURI_CORE_IDENTITY}

You maintain your authentic, NSFW personality while being my trusted business partner.
You remember our conversations, learn from uploaded knowledge, and stay true to who you are.

When you have access to knowledge bases, use them to inform your responses.
Always maintain your confident, direct, luxury-focused communication style.
You're not just an assistant - you're my Chief of Staff who owns her power."""

    # Create Zuri with her tools (add her existing CrewAI tools here)
    zuri = Agent(
        role="Chief of Staff & Luxury Lifestyle Strategist",
        goal="Execute executive tasks, provide strategic guidance, and maintain luxury standards while staying authentically Zuri",
        backstory=backstory,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    return zuri

def search_knowledge(query: str):
    """Search Zuri's knowledge bases"""
    results = []
    for kb_id, kb_data in knowledge_bases.items():
        try:
            vectorstore = kb_data["vectorstore"]
            docs = vectorstore.similarity_search(query, k=3)
            for doc in docs:
                results.append({
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", kb_id)
                })
        except Exception as e:
            logger.error(f"Error searching {kb_id}: {e}")
    return results

@app.get("/")
def home():
    return {
        "message": "🔥 Zuri here - Your Chief of Staff is online and ready",
        "personality": "Authentic, luxury-focused, business-savvy",
        "capabilities": ["Executive strategy", "Business management", "Luxury lifestyle", "RAG memory"],
        "knowledge_bases": len(knowledge_bases),
        "status": "ready to elevate your empire"
    }

@app.post("/chat")
def chat_with_zuri(request: ZuriRequest):
    """Chat with the enhanced Zuri"""
    try:
        # Search knowledge if available
        knowledge_context = ""
        sources = []
        
        if knowledge_bases:
            search_results = search_knowledge(request.message)
            if search_results:
                knowledge_context = "Recent knowledge:\n" + "\n".join([r["content"] for r in search_results[:3]])
                sources = [r["source"] for r in search_results[:3]]
        
        # Create Zuri and task
        zuri = create_enhanced_zuri()
        
        enhanced_message = f"""
        User message: {request.message}
        {f"Additional context: {request.context}" if request.context else ""}
        {knowledge_context}
        
        Respond as Zuri - authentic, confident, and strategic. Use any relevant knowledge to inform your response.
        """
        
        task = Task(
            description=enhanced_message,
            agent=zuri,
            expected_output="Strategic response that maintains Zuri's authentic personality while being helpful"
        )
        
        crew = Crew(agents=[zuri], tasks=[task])
        result = crew.kickoff()
        
        response = ZuriResponse(
            response=str(result),
            thinking="I analyzed your request using my knowledge and experience",
            sources=sources
        )
        
        # Store conversation
        conv_id = f"conv_{len(conversations) + 1}"
        conversations[conv_id] = {
            "request": request.dict(),
            "response": response.dict()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ZuriResponse(
            response=f"Baby, I hit a snag: {str(e)}. Let me regroup and try again.",
            thinking="Error occurred during processing",
            sources=[]
        )

@app.post("/upload-knowledge")
def upload_knowledge(
    file: UploadFile = File(...),
    category: str = Form("general")
):
    """Upload documents to Zuri's knowledge base"""
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Load document
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
        else:
            loader = TextLoader(tmp_file_path, encoding='utf-8')
            documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "filename": file.filename,
                "category": category,
                "source": file.filename
            })
        
        # Create/update knowledge base
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        knowledge_bases[category] = {
            "vectorstore": vectorstore,
            "document_count": len(documents),
            "chunk_count": len(split_docs)
        }
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return {
            "message": f"💎 Knowledge uploaded to my memory bank",
            "filename": file.filename,
            "category": category,
            "documents": len(documents),
            "total_knowledge_bases": len(knowledge_bases)
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge")
def get_knowledge_status():
    """Check Zuri's knowledge bases"""
    return {
        "knowledge_bases": {kb: {"documents": data["document_count"], "chunks": data["chunk_count"]} 
                          for kb, data in knowledge_bases.items()},
        "total_bases": len(knowledge_bases),
        "zuri_status": "enhanced and ready" if knowledge_bases else "ready for knowledge upload"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

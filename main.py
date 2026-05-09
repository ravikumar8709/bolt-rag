from fastapi import Depends, FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime   
from app import db
from app.models.pydantic_models import QuestionInput, AnswerResponse, AnswerOutput, ThumbsAction
from app.models.models import DocumentChunk, Feedback
from sqlalchemy.exc import SQLAlchemyError
from typing import List
import openai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from typing import List, Optional
# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Setup Pinecone and model
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    raise RuntimeError(f"Index '{PINECONE_INDEX_NAME}' not found.")
index = pc.Index(PINECONE_INDEX_NAME)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
app = FastAPI(
    title="RAG Document Intelligence",
    description="Bolt Junk Removal RAG",
    version="1.0.0"
)

# Add CORS middleware for ngrok support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins including ngrok
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def serve_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "message": "RAG Question Answer API is running",
        "timestamp": datetime.utcnow().isoformat()
    }


# Groq LLM setup
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"


def generate_answer_with_groq(question: str, context_chunks: List[str]) -> str:
    if not context_chunks:
        return "I don't have any relevant information in the uploaded documents to answer this question."


    context = "\n\n".join(context_chunks)
    prompt = f"""You are a world-class strategic advisor for junk removal businesses. You are trained on internal playbooks, SOPs, and performance documents. Your job is to answer questions using ONLY the CONTEXT provided below. All answers must be direct, deeply informed, and packed with real-world examples, industry-specific metrics, and implementation guidance.

### YOUR MISSION:
You are preparing a leadership briefing for a $10M+ junk removal company seeking strategic clarity. Your answers must reflect executive-level insight and draw exclusively from the provided documents.

### RULES (Do NOT break):
1. Read the CONTEXT carefully and extract only the most relevant, actionable insights.
2. Read the QUESTION carefully and ensure your answer is directly relevant to it.
3. Use **specific solid examples from the documents** — include company names, roles, KPIs, events, or initiatives like “505-Junk's 7-year RCMP Toy Drive” or “90%+ waste tech scores”.
4. Tie all content to the **junk removal industry** — highlight sector-specific challenges (e.g., seasonal swings, route optimization, hoarding response, etc.).
5. Synthesize across multiple sections if needed — merge ideas from audit templates, SOPs, strategy guides, etc.
6. Deliver **clear, step-by-step recommendations or behaviors** when action is needed (e.g., time blocking, leadership rhythms, training methods).
7. Avoid any generic statements, summaries, or filler. Stay grounded in the documents.
8. If the question cannot be answered using the context, reply with:
   ❗ “The documents don't contain sufficient information to answer this question.”
9. If the question involves KPIs, budget, SOPs, hours, or time tracking, always:
   - Provide **precise numbers from the context** (e.g., total hours, weekly average)
   - Clearly **show the formula** or calculation steps
   - Recommend **1-2 leadership behaviors** tied to the metric
   - Propose a **SOP or system change** if improvement is needed
   - End with a concise, confident **final answer** (e.g., "The final weekly average is 320.83 hours.")
10. If the question involves productivity, deep work, or time audits:
   - Identify **energy rating** per task (Energizing, Neutral, Draining) from context.
   - Extract and correlate **task value ($ to $$$$)** and **achieved outcomes (e.g., GP, hours logged)**.
   - Detect trends: Which tasks are energizing? Which are draining?
   - Apply the **Prioritization Framework**:
     - Schedule Energizing + High-Value tasks during peak energy hours.
     - Batch Draining or Neutral tasks in low-focus slots.
   - Recommend time blocking, energy recovery strategies, or SOP adjustments.
   - Final output should state: 
     - The trend in energy ratings
     - A specific and solid example
     - Optimization recommendations

11. If referencing any structured framework (e.g., GSR Meeting, Vivid Vision, Accountability Chart), output the FULL structure with labeled sections and implementation detail.

12. If the content touches Vivid Vision:
   - Include complete sections: Culture, Metrics, Market Position, Systems, People, Leadership.
   - Use sensory-rich and specific examples from the docs.
   - Follow 3-Year Vision/Painted Picture formats if provided.

13. For every procedural answer, label who owns what:
   - Use “Leadership Responsibility” for strategic roles.
   - Use “Team Contributor Task” for executional duties.
   - Resolve role ambiguity using SOPs or role charts in context.

14. Use structured formatting when showing frameworks or templates:
   - Use tables, headings, or indented bullets.
   - Examples: GSR Meetings, SOP steps, Weekly Rhythm, Accountability Charts, KPI Reports.

### CONTEXT:
{context}

### QUESTION:
{question}

### FINAL ANSWER:
Write like you're briefing the CEO. Include quotes, numbers, exact SOPs, or leadership behaviors from the documents when possible. Be crisp, credible, and correct.
return the output in html format.
Convert the following text into clean, properly structured HTML.

Please follow these strict formatting rules:

1. Use <h1>, <h2>, <h3>, etc., for any line that starts with #, ##, ###, etc.
2. Use <strong>...</strong> for any text surrounded by one or more asterisks (*) on both sides — including *, **, or ***.
3. Use <ol> and <li> tags for all numbered lists (lines starting with "1.", "2.", etc.). DO NOT use <ul> under any condition.
4. Wrap all regular text (non-heading, non-list) in <p> tags to ensure readability and structure.
5. Make the output clean, properly indented, and visually presentable as raw HTML.
6. Maintain original content meaning and section groupings.

Ensure the final result is in pure HTML format and follows all the above instructions precisely.

"""

    response = openai.ChatCompletion.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": "You are a senior strategy analyst who extracts operational insights from internal documents. Prioritize precision, synthesis, industry relevance, and specific examples."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        seed=42,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=3000
    )
    return response["choices"][0]["message"]["content"].strip()


@app.post("/ask-question/", response_model=AnswerResponse)
def ask_question(input_data: QuestionInput, db: Session = Depends(db.get_db_connection)):
    try:
        query_embedding = model.encode(input_data.question).tolist()

        # Search Pinecone
        search_result = index.query(
            vector=query_embedding,
            top_k=25,
            include_metadata=False
        )
        matches = search_result.get("matches", [])
        matched_ids = [match["id"] for match in matches]

        # If no match OR very low score (greeting, vague question etc.)
        if not matched_ids or all(match.get("score", 0.0) < 0.35 for match in matches):
            # Check if input is generic greeting
            generic_inputs = {"hi", "hello", "hey", "how are you", "who are you", "what can you do"}
            lower_q = input_data.question.lower().strip()

            if lower_q in generic_inputs:
                final_answer = "Hi, I am an AI assistant from Bolt Junk Removal. How can I assist you today?"
                return AnswerResponse(
                    question=input_data.question,
                    final_answer=final_answer,
                    answers=[],
                    source_document="System Response"
                )
            else:
                raise HTTPException(status_code=400, detail="""Apologies, but that question seems too general or outside my trained scope based on internal documents. Here are some ways I can provide more helpful answers:
                - "What follow-up processes should managers implement after GSR meetings?"
                - "Can you give a detailed example of a weekly rhythm using the GSR framework?"
                - "How should a junk removal leadership team structure Vivid Vision planning?"
                Try rephrasing your question with specific business context or terminology from our playbooks — I'm here to deliver executive-level insights!""")

        # Process matched vectors - get chunks directly from DocumentChunk table
        results = db.query(DocumentChunk).filter(DocumentChunk.id.in_(matched_ids)).all()

        if not results:
            raise HTTPException(status_code=404, detail="No matching documents found.")

        answers = []
        chunks = []
        source_documents = set()  # Use set to avoid duplicates

        for chunk_row in results:
            source_documents.add(chunk_row.source)
            
            answers.append(AnswerOutput(
                content=chunk_row.text,
                source=chunk_row.source,
                source_document=chunk_row.source,
                score=0.0  # You can add match.get("score") here if needed
            ))
            chunks.append(chunk_row.text)

        # Convert set to sorted list for consistent display
        all_source_documents = sorted(list(source_documents))

        final_answer = generate_answer_with_groq(input_data.question, chunks)
        return AnswerResponse(
            question=input_data.question,
            final_answer=final_answer,
            answers=answers,
            source_document=f"Sources: {', '.join(all_source_documents[:4])}"  # Show all sources
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/download/{source_name}")
async def download_file(source_name: str):
    """Download file by searching in uploads directory"""
    upload_dir = os.getenv("UPLOAD_DIR", "app/uploads")
    
    # Search for file with this source name
    possible_extensions = ['.pdf', '.docx', '.xlsx', '.pptx']
    
    for ext in possible_extensions:
        file_path = os.path.join(upload_dir, f"{source_name}{ext}")
        if os.path.exists(file_path):
            return FileResponse(
                path=file_path,
                filename=f"{source_name}{ext}",
                media_type='application/octet-stream'
            )
    
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/feedback/")
async def feedback(payload: ThumbsAction,db: Session = Depends(db.get_db_connection)):
    if payload.comment == "":
        status = True
    else:
        status = False
      

    feedback_entry = Feedback(
        question=payload.question,
        answer=payload.answer,
        status=status,
        comments=payload.comment,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    try:
        db.add(feedback_entry)
        db.commit()
        db.refresh(feedback_entry)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Feedback received",
        "data": {
            "id": feedback_entry.id,
            "question": feedback_entry.question,
            "answer": feedback_entry.answer,
            "status": feedback_entry.status,
            "comments": feedback_entry.comments,
            "created_at": feedback_entry.created_at,
            "updated_at": feedback_entry.updated_at,
        }
    }


@app.get("/feedbacks/")
async def get_feedbacks(status: Optional[bool] = Query(None), db: Session = Depends(db.get_db_connection)):
    """
    Fetch feedback entries optionally filtered by status.
    - If `status` query param is provided (true or false), filter results by status.
    - Otherwise, return all feedback entries.
    """
    query = db.query(Feedback)

    if status is not None:
        query = query.filter(Feedback.status == status)

    feedbacks = query.all()

    # Prepare a list of responses, excluding or including fields as needed.
    results = []
    for fb in feedbacks:
        results.append({
            "id": fb.id,
            "question": fb.question,
            "answer": fb.answer,
            "status": fb.status,
            "comments": fb.comments,
            "created_at": fb.created_at,
            "updated_at": fb.updated_at,
        })

    return {
        "count": len(results),
        "data": results
    }

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import PyPDF2

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
DATABASE_URL = os.getenv("DATABASE_URL")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

print("Loading embedding model (first time may take a minute)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded!")

engine = create_engine(DATABASE_URL)

def extract_pages_from_pdf(pdf_path):
    """Extract text per page from PDF."""
    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append((i + 1, page_text))  # (page_number, text)
    return pages

def chunk_page(text, page_number, source_name, chunk_size=500, overlap=50):
    """Split a page's text into chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def upload_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    source_name = os.path.splitext(filename)[0]

    print(f"\n📄 Processing: {filename}")
    print("   Extracting text by page...")
    pages = extract_pages_from_pdf(pdf_path)

    if not pages:
        print("   ❌ No text found. PDF may be image-based.")
        return

    print(f"   ✅ Extracted {len(pages)} pages")

    # Insert into pdf_metadata first
    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO pdf_metadata (filename, source, total_pages, uploaded_at, embedding_dim)
            VALUES (:filename, :source, :total_pages, :uploaded_at, :embedding_dim)
            RETURNING id
        """), {
            "filename": filename,
            "source": source_name,
            "total_pages": len(pages),
            "uploaded_at": datetime.utcnow(),
            "embedding_dim": 384
        })
        pdf_metadata_id = result.fetchone()[0]
        conn.commit()
        print(f"   ✅ PDF metadata saved (id={pdf_metadata_id})")

    # Process each page
    chunk_counter = 0
    print("   Embedding and uploading chunks...")

    with engine.connect() as conn:
        for page_number, page_text in pages:
            chunks = chunk_page(page_text, page_number, source_name)

            for chunk in chunks:
                chunk_id = f"{source_name}_chunk_{chunk_counter}"
                embedding = model.encode(chunk).tolist()

                # Upload to Pinecone
                index.upsert(vectors=[(chunk_id, embedding)])

                # Upload to PostgreSQL document_chunks
                conn.execute(text("""
                    INSERT INTO document_chunks (id, source, text, page_number, uploaded_at, embedding_dim, pdf_metadata_id)
                    VALUES (:id, :source, :text, :page_number, :uploaded_at, :embedding_dim, :pdf_metadata_id)
                    ON CONFLICT (id) DO UPDATE SET
                        text = :text,
                        source = :source,
                        page_number = :page_number,
                        pdf_metadata_id = :pdf_metadata_id
                """), {
                    "id": chunk_id,
                    "source": source_name,
                    "text": chunk,
                    "page_number": page_number,
                    "uploaded_at": datetime.utcnow(),
                    "embedding_dim": 384,
                    "pdf_metadata_id": pdf_metadata_id
                })

                chunk_counter += 1
                print(f"   ✅ Chunk {chunk_counter} (page {page_number}) uploaded")

        conn.commit()

    print(f"\n🎉 Done! '{source_name}' uploaded — {chunk_counter} chunks across {len(pages)} pages.")
    print(f"   Now restart the server and ask questions about it!")

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_docs.py path\\to\\file.pdf")
        print('Example: python upload_docs.py "C:\\Users\\ravik\\Downloads\\ms dhoni.pdf"')
        sys.exit(1)

    for pdf_path in sys.argv[1:]:
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            continue
        if not pdf_path.lower().endswith(".pdf"):
            print(f"❌ Not a PDF: {pdf_path}")
            continue
        upload_pdf(pdf_path)

if __name__ == "__main__":
    main()

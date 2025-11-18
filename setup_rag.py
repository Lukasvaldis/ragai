import os
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Kompatibelt import for Document (håndterer forskellige langchain-versioner)
try:
    from langchain.schema import Document
except Exception:
    try:
        from langchain.docstore.document import Document
    except Exception:
        # Enkel fallback Document hvis langchain ikke tilbyder klassen
        from dataclasses import dataclass
        @dataclass
        class Document:
            page_content: str
            metadata: dict = None

# --- Indstillinger ---
DATA_PATH = r"C:\mydev\ragai\rag_docs"
VECTOR_DB_PATH = r"C:\mydev\ragai\chroma_db"

def create_vector_db():
    print("Starter dataforberedelse...")
    print('*' * 40)

    documents = []
    if not os.path.exists(DATA_PATH):
        print(f"Fejl: DATA_PATH findes ikke: {DATA_PATH}")
        return

    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        lower = filename.lower()
        if lower.endswith(".pdf"):
            print(f"Indlæser PDF: {filename}")
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Kunne ikke indlæse PDF {filename}: {e}")
        elif lower.endswith(".csv"):
            print(f"Indlæser CSV: {filename}")
            try:
                # Læs som tekst (dtype=str) for at undgå dtype-problemer
                df = pd.read_csv(file_path, dtype=str, encoding="utf-8", on_bad_lines="skip")
                for i, row in df.iterrows():
                    # Sammensæt en læsbar tekst fra hele rækken (tilpas hvis du kun vil bruge en kolonne)
                    content = " | ".join(f"{col}: {row[col]}" for col in df.columns)
                    documents.append(Document(page_content=str(content), metadata={"source": filename, "row": int(i)}))
            except Exception as e:
                print(f"Kunne ikke læse CSV {filename}: {e}")
        elif lower.endswith(".txt"):
            print(f"Indlæser TXT: {filename}")
            try:
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
            except Exception as e:
                print(f"Kunne ikke læse TXT {filename}: {e}")
        else:
            # Ignorer andre filtyper
            continue

    print('*' * 40)

    if not documents:
        print(f"Fejl: Ingen understøttede filer fundet i mappen: {DATA_PATH}")
        return

    print(f"Antal dokumenter/sider/rækker indlæst: {len(documents)}")

    # Split i mindre bidder
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    print(f"Opdelt i {len(texts)} tekstbidder.")
    print('*' * 40)

    # Embeddings og gem i Chroma
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Fejl ved oprettelse af embeddings: {e}")
        return

    print("Opretter/udfylder vektordatabase...")
    try:
        db = Chroma.from_documents(texts, embeddings, persist_directory=VECTOR_DB_PATH)
        db.persist()
        print("✅ Vektordatabase oprettet og gemt.")
    except Exception as e:
        print(f"Fejl ved oprettelse af Chroma DB: {e}")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Oprettet mappen '{DATA_PATH}'. Læg dine CSV/TXT/PDF filer her og kør igen.")
    else:
        create_vector_db()

# Kør denne fil én gang for at oprette vektordatabase, FØR du kører Flask-applikationen.
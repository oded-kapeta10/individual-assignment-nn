import os
import time
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# CONFIGURATION
# ==========================================
# 1. API KEYS (Replace these with your actual keys)
LLMOD_API_KEY = "sk-d98cBXma0vKhK7FaB3VmGA"  # Your Key from LLMod
PINECONE_API_KEY = "pcsk_abmnu_KFUmKLFy68RxYi1Ur4gAewM9FRGUifEqiwnypmFKamXGHm1CpDymDztcUEHnTk3"  # Your Key from Pinecone

# 2. PINECONE SETTINGS
INDEX_NAME = "ted-rag"
NAMESPACE = "ns1"  # Optional, helps organize data

# 3. CHUNKING SETTINGS (Per assignment constraints)
# Constraint: Max 2048 tokens. We use 1000 to be safe and precise.
# Constraint: Max 30% overlap. We use 200 chars (~10-20%).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==========================================
# SETUP
# ==========================================

# 1. Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
# Check if index exists, if not create it (Dimensions MUST be 1536)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Required for text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# [cite_start]2. Initialize Embeddings via LLMod [cite: 410-415]
embeddings = OpenAIEmbeddings(
    model="RPRTHPB-text-embedding-3-small",
    openai_api_key=LLMOD_API_KEY,
    openai_api_base="https://api.llmod.ai/v1",  # The LLMod Proxy URL
)

# [cite_start]3. Initialize Text Splitter [cite: 533, 554]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Tries to split by paragraphs first
)


# ==========================================
# MAIN PIPELINE
# ==========================================

def process_and_upload():
    # Load your MINI dataset first to save budget!
    df = pd.read_csv("mini_dataset.csv")
    print(f"Loaded {len(df)} rows from dataset.")

    vectors_to_upsert = []
    batch_size = 50  # Upsert to Pinecone in batches

    for i, row in df.iterrows():
        talk_id = str(row['talk_id'])
        transcript = str(row['transcript'])

        # SKIP empty transcripts
        if not transcript or transcript.lower() == 'nan':
            continue

        # 1. CHUNK THE TRANSCRIPT
        chunks = text_splitter.split_text(transcript)
        print(f"Processing Talk {talk_id}: {row['title']} -> {len(chunks)} chunks")

        # 2. EMBED CHUNKS (Batch embedding saves time/calls)
        # LangChain handles the API loop for us
        try:
            chunk_embeddings = embeddings.embed_documents(chunks)
        except Exception as e:
            print(f"Error embedding talk {talk_id}: {e}")
            continue

        # 3. PREPARE VECTORS FOR PINECONE
        for j, (chunk_text, vector) in enumerate(zip(chunks, chunk_embeddings)):
            # Create a unique ID for this chunk
            chunk_id = f"{talk_id}_chunk_{j}"

            # [cite_start]Prepare Metadata (The "Context" for your LLM later) [cite: 8]
            metadata = {
                "talk_id": talk_id,
                "title": row['title'],
                "speaker": row['speaker_1'],
                "url": row['url'],
                "text": chunk_text  # IMPORTANT: Storing the text to retrieve later
            }

            vectors_to_upsert.append({
                "id": chunk_id,
                "values": vector,
                "metadata": metadata
            })

        # 4. UPSERT TO PINECONE (in batches)
        if len(vectors_to_upsert) >= batch_size:
            index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
            print(f"Upserted {len(vectors_to_upsert)} chunks to Pinecone...")
            vectors_to_upsert = []  # Reset list

    # Upsert any remaining vectors
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
        print(f"Upserted final {len(vectors_to_upsert)} chunks.")

    print("Pipeline Complete!")


if __name__ == "__main__":
    process_and_upload()
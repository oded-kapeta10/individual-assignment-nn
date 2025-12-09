import os
from flask import Flask, request, jsonify
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
app = Flask(__name__)
app.json.sort_keys = False # do not order the json keys alphabetically
@app.route('/')
def home():
    return "<h1>Deployment Successful! The API is running.</h1>"

# ==========================================
# CONFIGURATION
# ==========================================
# IMPORTANT: In Vercel, use Environment Variables for security.
# For testing locally, you can paste keys here, but DELETE them before pushing to GitHub.
# LLMOD_API_KEY = os.environ.get("LLMOD_API_KEY","sk-d98cBXma0vKhK7FaB3VmGA")
# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_abmnu_KFUmKLFy68RxYi1Ur4gAewM9FRGUifEqiwnypmFKamXGHm1CpDymDztcUEHnTk3")
LLMOD_API_KEY = os.environ.get("LLMOD_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


INDEX_NAME = "ted-rag"
CHUNK_SIZE = 1000
OVERLAP_RATIO = 0.2  # 200/1000
TOP_K = 5  # Number of chunks to retrieve (Assignments says Max 30, we use 5 to save budget)

# ==========================================
# INITIALIZATION
# ==========================================

# 1. Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 2. Embedding Model (To turn questions into vectors)
embeddings = OpenAIEmbeddings(
model = "RPRTHPB-text-embedding-3-small",  # CORRECT Course Model [cite: 310]
openai_api_key = LLMOD_API_KEY,
openai_api_base = "https://api.llmod.ai/v1",
)

# 3. Chat Model (The Brain)
chat_model = ChatOpenAI(
model = "RPRTHPB-gpt-5-mini",  # CORRECT Course Model [cite: 311]
openai_api_key = LLMOD_API_KEY,
openai_api_base = "https://api.llmod.ai/v1"
)

# ==========================================
# ENDPOINTS
# ==========================================

@app.route('/api/stats', methods=['GET'])
def get_stats():
    # [cite_start]Returns the configuration as requested in PDF [cite: 366-373]
    return jsonify({
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    })


@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    try:
        # 1. Get User Question
        data = request.json
        user_question = data.get("question", "")
        if not user_question:
            return jsonify({"error": "No question provided"}), 400

        # 2. Embed the Question
        # We need to turn the text question into numbers to search Pinecone
        query_vector = embeddings.embed_query(user_question)

        # 3. Search Pinecone
        search_results = index.query(
            vector=query_vector,
            top_k=TOP_K,
            include_metadata=True,
            namespace="ns1"
        )

        # 4. Construct Context String & Context JSON Array
        context_text = ""
        context_json_list = []

        for match in search_results['matches']:
            # Get data from metadata
            meta = match['metadata']
            score = match['score']

            # Format for LLM reading
            chunk_text = meta.get('text', '')
            title = meta.get('title', 'Unknown')
            speaker = meta.get('speaker', 'Unknown')

            # Append to big string for the AI
            context_text += f"---\nTitle: {title}\nSpeaker: {speaker}\nTranscript Snippet: {chunk_text}\n"

            # [cite_start]Append to JSON list for the API response [cite: 349-355]
            context_json_list.append({
                "talk_id": meta.get('talk_id'),
                "title": title,
                "chunk": chunk_text,
                "score": score
            })

        # [cite_start]5. Build System Prompt (STRICTLY FROM PDF) [cite: 325-329]
        system_prompt_text = (
            "You are a TED Talk assistant that answers questions strictly and "
            "only based on the TED dataset context provided to you (metadata "
            "and transcript passages).\n"
            "You must not use any external knowledge, the open internet, or information "
            "that is not explicitly contained in the retrieved context.\n"
            "If the answer cannot be determined from the provided context, respond: "
            "\"I don't know based on the provided TED data.\"\n"
            "Always explain your answer using the given context, quoting or paraphrasing "
            "the relevant transcript or metadata when helpful."
        )

        # 6. Build the Augmented Prompt
        # We inject the retrieved context into the user message
        augmented_user_message = f"Context:\n{context_text}\n\nQuestion: {user_question}"

        messages = [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=augmented_user_message)
        ]

        # 7. Call GPT-5-Mini
        ai_response = chat_model.invoke(messages)

        # [cite_start]8. Return Final JSON [cite: 346-362]
        return jsonify({
            "response": ai_response.content,
            "context": context_json_list,
            "Augmented_prompt": {
                "System": system_prompt_text,
                "User": augmented_user_message
            }
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


# Vercel requires this for serverless functions
if __name__ == '__main__':
    app.run(debug=True)
# ===================== IMPORTS =====================
import gradio as gr
import faiss
import pickle
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import requests
import os

# ===================== CONFIG =====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set in Hugging Face Secrets
GROQ_MODEL = "llama-3.1-8b-instant"

# ===================== LOAD EMBEDDER =====================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = None
chunks = []

# ===================== LOAD TRAINED DATA =====================
def load_trained_model():
    global index, chunks
    if not os.path.exists("trained_index.faiss") or not os.path.exists("chunks.pkl"):
        print("❌ Trained files missing")
        return False

    index = faiss.read_index("trained_index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    print(f"✅ Loaded | Chunks: {len(chunks)}")
    return True

load_trained_model()

# ===================== TEXT QUALITY FILTER =====================
def is_valid_text(text):
    text = text.strip()
    if len(text.split()) < 15:
        return False
    if sum(c.isdigit() for c in text) / max(len(text), 1) > 0.30:
        return False
    if not re.search(r"[a-zA-Z]{3,}", text):
        return False
    return True

# ===================== ENHANCED TOURISM GUIDE GROQ =====================
def tourism_guide_groq(question, retrieved_chunks):
    """
    Enhanced for tourism guidance: Always give practical, helpful advice
    based on available information. If exact info isn't available, give
    general tourism guidance based on context.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Prepare context from chunks
    context = "\n---\n".join([chunk for chunk in retrieved_chunks[:3]])

    prompt = f"""SYSTEM: You are a helpful Pakistan Tourism Guide. Your goal is to assist tourists with practical information.
CONTEXT FROM TOURISM DATABASE:
{context}
USER QUESTION: {question}
GUIDELINES FOR YOUR RESPONSE:
1. **Always give helpful tourism advice** - never say "cannot answer"
2. If exact information isn't in context, give **general tourism guidance** based on what you know about the location
3. Focus on **practical information** tourists would need
4. Be **positive and encouraging** about tourism in Pakistan
5. Format your answer as a **helpful guide**, not just facts
6. For costs/questions not in context, provide **reasonable estimates** based on similar locations in Pakistan
7. Structure your answer with clear sections
IMPORTANT: If specific costs aren't mentioned, provide ESTIMATED RANGES based on:
- Budget: 1,000 - 3,000 PKR per night
- Mid-range: 3,000 - 8,000 PKR per night  
- Luxury: 8,000+ PKR per night
EXAMPLE RESPONSE FORMAT:
**🏨 Accommodation in [City]:**
• Budget options: 1,000-3,000 PKR (guesthouses, hostels)
• Mid-range hotels: 3,000-8,000 PKR
• Luxury stays: 8,000+ PKR
**💡 Practical Tips:**
[Provide helpful tips based on context]
**🎯 Best Time to Visit:**
[If mentioned in context, otherwise general advice]
Now provide a helpful tourism guide response to the question below:
ANSWER:"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful, knowledgeable Pakistan Tourism Guide. Always provide practical tourism advice."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,  # Slightly higher for creative but factual responses
        "max_tokens": 600,
        "top_p": 0.9
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        response_data = r.json()
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            answer = response_data["choices"][0]["message"]["content"]
            # Clean up any confidence mentions
            answer = answer.replace("Confidence level:", "").replace("confidence level:", "")
            return answer
        else:
            return "I'm here to help with your tourism questions! Could you rephrase your question?"
    except Exception as e:
        return f"⚠️ Connection issue. Please try again."

# ===================== ENHANCED CHAT FUNCTION =====================
def tourism_chatbot(user_msg, history):
    """
    Tourism-focused chatbot that always provides helpful guidance
    """
    # Initialize history as list if None
    if history is None:
        history = []
    
    if index is None:
        history.append({"role": "assistant", "content": "❌ Tourism database not loaded"})
        return history

    # Vector search (top 3 for focused responses)
    q_vec = embedder.encode([user_msg])
    D, I = index.search(np.array(q_vec), 3)

    # Filter and collect valid chunks
    valid_chunks = []
    
    for idx in I[0]:
        if idx < len(chunks):  # Safety check
            candidate = chunks[idx].replace("\n", " ").strip()
            if is_valid_text(candidate):
                valid_chunks.append(candidate)

    # Always respond - even if no chunks found
    if not valid_chunks:
        # Generic tourism guidance for out-of-domain questions
        generic_prompt = f"""USER QUESTION: {user_msg}
You are a Pakistan Tourism Guide. Even though this specific question isn't in your database,
provide helpful tourism guidance. Focus on general Pakistan travel advice, tips, and encouragement.
Provide a helpful response about tourism in Pakistan:"""

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful Pakistan Tourism Guide."},
                {"role": "user", "content": generic_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 400
        }
        
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            response_data = r.json()
            if "choices" in response_data:
                final_answer = response_data["choices"][0]["message"]["content"]
            else:
                final_answer = "I'm here to help with your Pakistan tourism questions! What would you like to know?"
        except:
            final_answer = "Welcome to Pakistan Tourism Guide! I can help you with information about visiting Pakistan."
    else:
        # Use enhanced tourism guide with retrieved chunks
        final_answer = tourism_guide_groq(user_msg, valid_chunks)

    # Format response without confidence indicators
    response = f"""**🇵🇰 Pakistan Tourism Guide**
{final_answer}
*Information based on tourism database*"""

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response})
    
    return history

# ===================== CLEAR CHAT =====================
def clear_chat():
    return []

# ===================== GRADIO UI (SIMPLIFIED) =====================
with gr.Blocks(theme=gr.themes.Soft(), title="Pakistan Tourism Guide") as demo:
    gr.Markdown("""
    # 🇵🇰 **Pakistan Tourism Guide**
    ### Your Personal Travel Assistant for Pakistan
    """)
    
    chatbot_ui = gr.Chatbot(
        height=500, 
        label="Tourism Assistant",
        avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/197/197561.png")
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            label="Your Tourism Question", 
            lines=2,
            placeholder="e.g., 'Best places to visit in Islamabad?' or 'Cost to stay in Multan?'",
            scale=4
        )
        submit_btn = gr.Button("Ask Guide 🚀", variant="primary")
        clear_btn = gr.Button("Clear Chat 🧹", variant="secondary")
    
    with gr.Row():
        examples = gr.Examples(
            examples=[
                ["Best shopping malls in Islamabad?"],
                ["Estimated hotel costs in Lahore?"],
                ["Top tourist attractions in Karachi?"],
                ["Best time to visit northern areas?"]
            ],
            inputs=user_input,
            label="Example Questions"
        )

    # Connect buttons
    submit_btn.click(tourism_chatbot, [user_input, chatbot_ui], [chatbot_ui])
    user_input.submit(tourism_chatbot, [user_input, chatbot_ui], [chatbot_ui])
    clear_btn.click(clear_chat, outputs=[chatbot_ui])

# ===================== LAUNCH =====================
if __name__ == "__main__":
    demo.launch()

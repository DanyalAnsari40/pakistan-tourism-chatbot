# app.py - Optimized for GitHub + Hugging Face Deployment

import os
import gradio as gr
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PakistanTourismBot:
    def __init__(self):
        self.model_loaded = False
        self.vector_db = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Lazy loading of model to save memory"""
        try:
            # Try to load the vector database
            if os.path.exists("trained_model"):
                from langchain_huggingface import HuggingFaceEmbeddings
                from langchain_community.vectorstores import FAISS
                
                logger.info("Loading embeddings...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                logger.info("Loading FAISS vector store...")
                self.vector_db = FAISS.load_local(
                    "trained_model",
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                self.model_loaded = True
                logger.info("✅ Model loaded successfully")
            else:
                logger.warning("⚠️ No trained model found. Using fallback responses.")
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self.model_loaded = False
    
    def ask(self, question):
        """Main query function"""
        question = question.strip().lower()
        
        # Greetings
        if any(word in question for word in ['hi', 'hello', 'hey']):
            return self._greeting_response()
        
        # Help
        if 'help' in question:
            return self._help_response()
        
        # If model is loaded, use it
        if self.model_loaded and self.vector_db:
            try:
                docs = self.vector_db.similarity_search(question, k=3)
                if docs:
                    return self._format_from_docs(question, docs)
            except Exception as e:
                logger.error(f"Search error: {e}")
        
        # Fallback responses
        return self._fallback_response(question)
    
    def _greeting_response(self):
        return """🇵🇰 **Welcome to Pakistan Tourism Chatbot!**

I can help you with:
• Tourist destinations in Pakistan
• Best times to visit
• Attractions and sightseeing
• Travel tips and information
• Local culture and food

What would you like to know?"""
    
    def _help_response(self):
        return """**Sample Questions:**

🏔️ **Destinations:**
- Best places to visit in Pakistan?
- Top tourist spots in Lahore?
- Must-see places in Northern Areas?

📅 **Timing:**
- When to visit Hunza Valley?
- Best season for Murree?
- Weather in Islamabad?

🏞️ **Attractions:**
- What to see in Karachi?
- Historical sites in Pakistan?
- Adventure activities?

🍽️ **Food & Culture:**
- Local food to try?
- Cultural experiences?
- Shopping recommendations?

🚗 **Travel:**
- How to reach Skardu?
- Accommodation options?
- Travel tips for Pakistan?"""
    
    def _format_from_docs(self, question, docs):
        """Format response from documents"""
        context = "\n".join([doc.page_content[:500] for doc in docs])
        
        # Simple formatting based on question type
        if any(word in question for word in ['best place', 'where to go', 'recommend']):
            return f"**Top Recommendations:**\n\n{context[:800]}..."
        
        elif any(word in question for word in ['best time', 'when to visit', 'season']):
            return f"**Best Time Information:**\n\n{context[:600]}..."
        
        elif any(word in question for word in ['attraction', 'what to see', 'things to do']):
            return f"**Attractions:**\n\n{context[:700]}..."
        
        else:
            return f"**Information:**\n\n{context[:500]}..."
    
    def _fallback_response(self, question):
        """Fallback responses when model not available"""
        responses = {
            'lahore': """**Lahore - Cultural Heart:**
• Badshahi Mosque (Mughal architecture)
• Lahore Fort (UNESCO World Heritage)
• Food Street (Local cuisine)
• Shalimar Gardens (Mughal gardens)
• Best Time: October to March""",
            
            'islamabad': """**Islamabad - Capital City:**
• Faisal Mosque (Modern Islamic design)
• Daman-e-Koh (Margalla Hills view)
• Lok Virsa Museum (Cultural heritage)
• Best Time: March to October""",
            
            'hunza': """**Hunza Valley - Mountain Paradise:**
• Attabad Lake (Turquoise mountain lake)
• Altit Fort (Ancient fortification)
• Passu Cones (Unique peaks)
• Best Time: May to October""",
            
            'karachi': """**Karachi - Coastal Metropolis:**
• Clifton Beach (Popular seaside)
• Quaid's Mausoleum (Founder's tomb)
• Dolmen Mall (Shopping)
• Best Time: November to February""",
            
            'murree': """**Murree - Hill Station:**
• Mall Road (Shopping and dining)
• Pindi Point (Scenic view)
• Kashmir Point (Panoramic view)
• Best Time: May to September"""
        }
        
        for key, response in responses.items():
            if key in question:
                return response
        
        return """I can provide information about Pakistan tourism. Try asking about:

• Specific cities (Lahore, Islamabad, Karachi, Hunza)
• Tourist attractions
• Best times to visit
• Travel information
• Local food and culture

Example: "Best places in Lahore?" or "When to visit Hunza?""""

# Create chatbot instance
chatbot = PakistanTourismBot()

# Gradio Interface
def create_interface():
    with gr.Blocks(
        title="Pakistan Tourism Chatbot",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {max-width: 800px; margin: auto;}
        footer {visibility: hidden;}
        """
    ) as demo:
        
        gr.Markdown("""
        # 🏔️ Pakistan Tourism Chatbot
        ### Domain-specific AI assistant trained on Pakistan tourism data
        
        🇵🇰 Ask me about tourist destinations, attractions, travel tips, and cultural information!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot_interface = gr.Chatbot(height=400, label="Chat")
                msg = gr.Textbox(
                    placeholder="Ask about Pakistan tourism...",
                    label="Your Question"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Ask", variant="primary")
                    clear_btn = gr.Button("Clear")
            
            with gr.Column(scale=1):
                gr.Markdown("### 💡 Quick Questions")
                
                questions = [
                    "Best places in Pakistan?",
                    "When to visit Northern Areas?",
                    "Attractions in Lahore",
                    "Local food to try",
                    "Hill stations",
                    "Historical sites",
                    "Travel costs",
                    "How to reach Skardu?"
                ]
                
                for q in questions:
                    gr.Button(q, size="sm").click(
                        lambda q=q: q,
                        outputs=[msg]
                    ).then(
                        lambda q, history: (q, history + [[q, None]]),
                        inputs=[msg, chatbot_interface],
                        outputs=[msg, chatbot_interface]
                    ).then(
                        lambda q: chatbot.ask(q),
                        inputs=[msg],
                        outputs=[chatbot_interface]
                    )
                
                gr.Markdown("---")
                gr.Markdown("### 📊 System Status")
                status = "✅ Model Loaded" if chatbot.model_loaded else "⚠️ Using Fallback Mode"
                gr.Markdown(f"**Status:** {status}")
        
        # Event handlers
        def respond(message, history):
            response = chatbot.ask(message)
            history.append((message, response))
            return "", history
        
        msg.submit(respond, [msg, chatbot_interface], [msg, chatbot_interface])
        submit_btn.click(respond, [msg, chatbot_interface], [msg, chatbot_interface])
        clear_btn.click(lambda: None, None, chatbot_interface)
        
        gr.Markdown("---")
        gr.Markdown("*Trained on Pakistan tourism PDFs and CSV data. For latest information, check official sources.*")
    
    return demo

# Launch application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
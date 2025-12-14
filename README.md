# 🏔️ Pakistan Tourism Chatbot

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/your-username/pakistan-tourism-chatbot)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A domain-specific chatbot trained on Pakistan tourism PDFs and CSV data. This AI assistant provides information about tourist destinations, attractions, travel tips, and cultural information for Pakistan.

## ✨ Live Demo

**Try it here:** [https://huggingface.co/spaces/your-username/pakistan-tourism-chatbot](https://huggingface.co/spaces/your-username/pakistan-tourism-chatbot)

## 📋 Features

- **Domain-Specific**: Trained exclusively on Pakistan tourism data
- **Multi-format Support**: Processes both PDF and CSV files
- **Smart Responses**: Categorized answers for destinations, timing, food, etc.
- **User-Friendly Interface**: Clean Gradio web interface
- **Real-time Responses**: Instant answers to tourism queries
- **Fallback System**: Works even without model files

## 🏗️ Architecture

```mermaid
graph LR
    A[PDF/CSV Files] --> B[Text Processing]
    B --> C[Vector Database]
    C --> D[Similarity Search]
    E[User Query] --> D
    D --> F[Context Retrieval]
    F --> G[Response Generation]
    G --> H[Formatted Answer]
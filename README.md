📄 Adaptive RAG LLM Agent – README
Overview

This project implements an Adaptive Retrieval-Augmented Generation (RAG) agent using LangChain, LangGraph, and LLMs.
The agent answers user questions by intelligently deciding whether to use local documents (PDF) or web search.

🔧 How It Works

Loads and splits a PDF document (hr_manual.pdf)

Converts document chunks into vector embeddings using Ollama embeddings

Stores embeddings in an in-memory vector store

Enhances user queries for better retrieval

Retrieves relevant document context

Validates whether the context is sufficient

If relevant → answers using document context

If not relevant → performs web search (Tavily) and answers using web result

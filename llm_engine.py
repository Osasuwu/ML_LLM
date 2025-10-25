"""LLM engine using Google Gemini API."""
import google.generativeai as genai
from typing import List, Dict, Optional
import re


class LLMEngine:
    """Manage interactions with Google Gemini API."""
    
    def __init__(self, api_key: str, model_name: str = 'gemini-pro',
                 temperature: float = 0.3, max_output_tokens: int = 2048):
        """Initialize the LLM engine.
        
        Args:
            api_key: Google API key
            model_name: Name of the Gemini model to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_output_tokens: Maximum number of tokens in the response
        """
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        
        # Generation config
        self.generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_output_tokens,
        }
    
    def generate_answer(self, query: str, context_chunks: List[Dict],
                       use_context: bool = True) -> str:
        """Generate an answer to a query using retrieved context.
        
        Args:
            query: User's question
            context_chunks: List of relevant document chunks
            use_context: Whether to use retrieved context
            
        Returns:
            Generated answer
        """
        # Detect if query is Russian (presence of Cyrillic characters)
        is_russian = bool(re.search(r'[А-Яа-яЁё]', query))
        language_instruction = (
            "Отвечай на том же языке, что и вопрос. Если вопрос на русском — отвечай по-русски. "
            "Если в контексте недостаточно информации, прямо напиши, что данных недостаточно, и не выдумывай." if is_russian else
            "Answer strictly in the same language as the question. If the context is insufficient, say so clearly and do not hallucinate."
        )

        if use_context and context_chunks:
            # Build context from retrieved chunks
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                source = chunk['metadata'].get('filename', 'Unknown')
                content = chunk['content']
                context_parts.append(f"[Source {i}: {source}]\n{content}")
            
            context_text = "\n\n".join(context_parts)
            
            # Create prompt with context
            prompt = (
                f"You are a helpful AI assistant for a corporate knowledge base system.\n"
                f"{language_instruction}\n"
                f"Carefully analyze ALL provided context from internal documents.\n"
                f"Pay special attention to specific numbers, metrics, dates, and quantitative data.\n"
                f"If the answer is in the context, provide it with exact values and cite sources like [Source 1: filename].\n"
                f"If the information is truly not in the context, clearly state that.\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"QUESTION: {query}\n\n"
                f"ANSWER:"
            )
        else:
            # Direct query without context
            prompt = (
                f"You are a helpful AI assistant. {language_instruction}\n\n"
                f"QUESTION: {query}\n\nANSWER:"
            )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_summary(self, text: str, max_length: int = 500) -> str:
        """Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        prompt = f"""Please provide a concise summary of the following text in approximately {max_length} characters:

{text}

SUMMARY:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"

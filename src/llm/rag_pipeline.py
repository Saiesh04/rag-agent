"""
LLM integration and RAG pipeline implementation using Google Gemini
"""
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import re
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import time

logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG pipeline for question answering with context retrieval using Google Gemini"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        self.model = None
        self.use_gemini = True
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Google Gemini model"""
        try:
            # Get API key from environment
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key or api_key == 'your_google_gemini_api_key_here':
                logger.warning("Google API key not found. Using fallback mode.")
                self.use_gemini = False
                self.model = None
                return
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1024,
                )
            )
            
            logger.info(f"Google Gemini model '{self.model_name}' initialized successfully")
            self.use_gemini = True
            
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            logger.info("Falling back to simple text-based responses")
            self.use_gemini = False
            self.model = None
    
    def _generate_gemini_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Generate answer using Google Gemini"""
        try:
            # Prepare context text
            context_parts = []
            for i, ctx in enumerate(contexts[:3], 1):  # Use top 3 contexts
                source_info = f"[Source: {ctx.get('filename', 'Unknown')}"
                if ctx.get('chunk_id') is not None:
                    source_info += f", Section {ctx['chunk_id'] + 1}"
                source_info += "]"
                
                context_parts.append(f"Context {i}: {ctx['text']}\n{source_info}")
            
            context_text = "\n\n".join(context_parts)
            
            # Create optimized prompt for Gemini
            prompt = f"""You are a helpful AI assistant. Based on the provided context information, answer the user's question accurately and comprehensively.

Context Information:
{context_text}

Question: {query}

Instructions:
- Answer based ONLY on the information provided in the contexts above
- Be specific and cite relevant details from the contexts
- If the contexts don't contain sufficient information, clearly state that
- Provide a clear, well-structured answer
- Keep the response focused and relevant

Answer:"""
            
            # Generate response using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 500,
                }
            )
            
            answer = response.text.strip()
            
            # Add source information
            source_files = list(set([ctx.get('filename', 'Unknown') for ctx in contexts[:3]]))
            if len(source_files) > 0:
                answer += f"\n\n*Sources: {', '.join(source_files)}*"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {str(e)}")
            return self._generate_fallback_answer(query, contexts)
    
    def _generate_fallback_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Generate a simple fallback answer when Gemini is not available"""
        if not contexts:
            return "I don't have enough information to answer your question. Please try uploading relevant documents."
        
        # Use the most relevant context
        best_context = contexts[0]
        context_text = best_context['text']
        
        # Simple extractive approach - find sentences that might contain the answer
        sentences = context_text.split('.')
        relevant_sentences = []
        
        query_words = set(query.lower().split())
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                relevant_sentences.append((sentence.strip(), overlap))
        
        if relevant_sentences:
            # Sort by word overlap and take the best ones
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            answer = '. '.join([s[0] for s in relevant_sentences[:2]])
            return f"{answer}."
        
        # If no good sentences found, return first part of context
        return context_text[:200] + "..."
    
    def rerank_contexts(self, query: str, contexts: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Re-rank retrieved contexts using BM25"""
        try:
            if not contexts:
                return []
            
            # Extract texts for BM25
            corpus = [ctx['text'] for ctx in contexts]
            
            # Tokenize for BM25
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            tokenized_query = query.lower().split()
            
            # Initialize BM25
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Get BM25 scores
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Combine with vector similarity scores
            for i, ctx in enumerate(contexts):
                ctx['bm25_score'] = float(bm25_scores[i])
                # Combined score: weighted average of vector similarity and BM25
                ctx['combined_score'] = 0.7 * ctx['score'] + 0.3 * ctx['bm25_score']
            
            # Sort by combined score
            reranked = sorted(contexts, key=lambda x: x['combined_score'], reverse=True)
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Error in re-ranking: {str(e)}")
            # Return original contexts if re-ranking fails
            return contexts[:top_k]
    
    def build_context_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Build a prompt with context for the LLM"""
        if not contexts:
            return f"Question: {query}\n\nAnswer: I don't have enough information to answer this question."
        
        # Build context section
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            source_info = f"[Source: {ctx['filename']}"
            if ctx.get('chunk_id') is not None:
                source_info += f", Section {ctx['chunk_id'] + 1}"
            source_info += "]"
            
            context_parts.append(f"Context {i}: {ctx['text']}\n{source_info}")
        
        context_text = "\n\n".join(context_parts)
        
        # Build the full prompt
        prompt = f"""Based on the following context information, please answer the question. If the context doesn't contain enough information to answer the question, please say so.

Context Information:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an answer using Google Gemini with retrieved contexts"""
        try:
            # Re-rank contexts for better relevance
            reranked_contexts = self.rerank_contexts(query, contexts)
            
            if not reranked_contexts:
                return {
                    "answer": "I don't have enough relevant information to answer your question. Please try uploading relevant documents.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Generate answer using Gemini or fallback
            if self.use_gemini and self.model:
                answer = self._generate_gemini_answer(query, reranked_contexts)
                confidence = 0.85
            else:
                answer = self._generate_fallback_answer(query, reranked_contexts)
                confidence = 0.4
            
            # Calculate confidence based on context relevance
            avg_score = sum(ctx.get('score', 0.0) for ctx in reranked_contexts) / len(reranked_contexts)
            confidence = min(confidence * (avg_score + 0.3), 1.0)  # Adjust based on context quality
            
            # Prepare sources
            sources = []
            for ctx in reranked_contexts:
                sources.append({
                    "filename": ctx.get('filename', 'Unknown'),
                    "chunk_id": ctx.get('chunk_id', 0),
                    "text": ctx['text'][:300] + "..." if len(ctx['text']) > 300 else ctx['text'],
                    "score": ctx.get('score', 0.0),
                    "start_pos": ctx.get('start_pos', 0),
                    "end_pos": ctx.get('end_pos', len(ctx['text']))
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    

    
    def summarize_document(self, document_data: Dict[str, Any]) -> str:
        """Generate a summary of a document using Gemini"""
        try:
            text = document_data.get('text', '')
            if not text:
                return "No content available for summarization."
            
            # Truncate if too long for Gemini
            max_length = 3000  # Gemini can handle more text
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            if self.use_gemini and self.model:
                try:
                    prompt = f"""Please provide a comprehensive but concise summary of the following document. Focus on the main points, key concepts, and important details.

Document Content:
{text}

Summary:"""
                    
                    response = self.model.generate_content(
                        prompt,
                        generation_config={
                            'temperature': 0.5,
                            'top_p': 0.8,
                            'top_k': 40,
                            'max_output_tokens': 300,
                        }
                    )
                    
                    return response.text.strip()
                    
                except Exception as e:
                    logger.error(f"Error in Gemini summarization: {str(e)}")
                    return self._generate_extractive_summary(text)
            else:
                # Fallback summarization
                return self._generate_extractive_summary(text)
            
        except Exception as e:
            logger.error(f"Error in document summarization: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _generate_extractive_summary(self, text: str) -> str:
        """Generate an extractive summary as fallback"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return "No substantial content found for summarization."
        
        # Take first few sentences as summary
        summary_sentences = sentences[:3]
        return ". ".join(summary_sentences) + "."
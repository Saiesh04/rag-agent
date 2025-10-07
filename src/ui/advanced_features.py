"""
Advanced features for RAG Q&A System including source highlighting,
conversation management, and export functionality
"""
import json
import csv
from io import StringIO, BytesIO
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import re

class AdvancedFeatures:
    """Advanced features for enhanced user experience"""
    
    def __init__(self):
        pass
    
    def highlight_text_sources(self, text: str, sources: List[Dict[str, Any]], query: str) -> str:
        """Highlight relevant parts of the text based on sources and query"""
        try:
            # Extract keywords from the query
            query_words = set(word.lower().strip('.,!?;:') for word in query.split())
            
            highlighted_text = text
            
            # Highlight query terms
            for word in query_words:
                if len(word) > 2:  # Skip very short words
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    highlighted_text = pattern.sub(
                        f'<mark style="background-color: #ffeb3b; padding: 2px;">{word}</mark>',
                        highlighted_text
                    )
            
            return highlighted_text
            
        except Exception as e:
            return text  # Return original text if highlighting fails
    
    def create_source_attribution_map(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a detailed source attribution map"""
        attribution_map = {
            "total_sources": len(sources),
            "sources_by_file": {},
            "confidence_scores": [],
            "file_types": set(),
            "total_characters": 0
        }
        
        for source in sources:
            filename = source.get("filename", "Unknown")
            file_type = source.get("file_type", "unknown")
            score = source.get("score", 0)
            text_length = len(source.get("text", ""))
            
            attribution_map["confidence_scores"].append(score)
            attribution_map["file_types"].add(file_type)
            attribution_map["total_characters"] += text_length
            
            if filename not in attribution_map["sources_by_file"]:
                attribution_map["sources_by_file"][filename] = {
                    "chunks": 0,
                    "total_score": 0,
                    "file_type": file_type,
                    "characters": 0
                }
            
            attribution_map["sources_by_file"][filename]["chunks"] += 1
            attribution_map["sources_by_file"][filename]["total_score"] += score
            attribution_map["sources_by_file"][filename]["characters"] += text_length
        
        # Calculate average scores
        for file_info in attribution_map["sources_by_file"].values():
            file_info["avg_score"] = file_info["total_score"] / file_info["chunks"]
        
        attribution_map["file_types"] = list(attribution_map["file_types"])
        attribution_map["avg_confidence"] = sum(attribution_map["confidence_scores"]) / len(attribution_map["confidence_scores"]) if attribution_map["confidence_scores"] else 0
        
        return attribution_map
    
    def export_conversation_history(self, chat_history: List[Dict[str, Any]], format_type: str = "json") -> str:
        """Export conversation history in various formats"""
        try:
            if format_type.lower() == "json":
                return json.dumps(chat_history, indent=2, default=str)
            
            elif format_type.lower() == "csv":
                # Flatten the data for CSV
                flattened_data = []
                for chat in chat_history:
                    row = {
                        "timestamp": chat.get("timestamp", ""),
                        "question": chat.get("question", ""),
                        "answer": chat.get("answer", ""),
                        "confidence": chat.get("confidence", 0),
                        "num_sources": len(chat.get("sources", []))
                    }
                    
                    # Add source information
                    sources = chat.get("sources", [])
                    for i, source in enumerate(sources[:3]):  # First 3 sources
                        row[f"source_{i+1}_file"] = source.get("filename", "")
                        row[f"source_{i+1}_score"] = source.get("score", 0)
                    
                    flattened_data.append(row)
                
                # Convert to CSV
                output = StringIO()
                if flattened_data:
                    writer = csv.DictWriter(output, fieldnames=flattened_data[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened_data)
                
                return output.getvalue()
            
            elif format_type.lower() == "txt":
                # Plain text format
                text_output = []
                text_output.append("RAG Q&A System - Conversation History")
                text_output.append("=" * 50)
                text_output.append("")
                
                for i, chat in enumerate(chat_history, 1):
                    text_output.append(f"Conversation {i}")
                    text_output.append("-" * 20)
                    text_output.append(f"Timestamp: {chat.get('timestamp', 'N/A')}")
                    text_output.append(f"Question: {chat.get('question', 'N/A')}")
                    text_output.append(f"Answer: {chat.get('answer', 'N/A')}")
                    text_output.append(f"Confidence: {chat.get('confidence', 0):.2%}")
                    
                    sources = chat.get("sources", [])
                    if sources:
                        text_output.append("Sources:")
                        for j, source in enumerate(sources, 1):
                            text_output.append(f"  {j}. {source.get('filename', 'Unknown')} (Score: {source.get('score', 0):.3f})")
                    
                    text_output.append("")
                
                return "\n".join(text_output)
            
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            return f"Error exporting conversation history: {str(e)}"
    
    def generate_conversation_summary(self, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of the conversation history"""
        if not chat_history:
            return {"error": "No conversation history available"}
        
        summary = {
            "total_conversations": len(chat_history),
            "date_range": {
                "first": chat_history[0].get("timestamp", ""),
                "last": chat_history[-1].get("timestamp", "")
            },
            "average_confidence": 0,
            "top_sources": {},
            "question_topics": [],
            "performance_metrics": {}
        }
        
        # Calculate metrics
        confidences = []
        source_files = {}
        
        for chat in chat_history:
            confidence = chat.get("confidence", 0)
            confidences.append(confidence)
            
            # Count source files
            for source in chat.get("sources", []):
                filename = source.get("filename", "Unknown")
                if filename in source_files:
                    source_files[filename] += 1
                else:
                    source_files[filename] = 1
        
        # Calculate averages and top sources
        summary["average_confidence"] = sum(confidences) / len(confidences) if confidences else 0
        summary["top_sources"] = dict(sorted(source_files.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Performance metrics
        summary["performance_metrics"] = {
            "high_confidence_responses": sum(1 for c in confidences if c > 0.8),
            "medium_confidence_responses": sum(1 for c in confidences if 0.5 <= c <= 0.8),
            "low_confidence_responses": sum(1 for c in confidences if c < 0.5),
            "avg_sources_per_response": sum(len(chat.get("sources", [])) for chat in chat_history) / len(chat_history)
        }
        
        return summary
    
    def create_document_summary_report(self, processed_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a comprehensive summary report of processed documents"""
        if not processed_documents:
            return {"error": "No processed documents available"}
        
        report = {
            "total_documents": len(processed_documents),
            "total_characters": sum(doc.get("size", 0) for doc in processed_documents),
            "total_chunks": sum(doc.get("num_chunks", 0) for doc in processed_documents),
            "file_types": {},
            "processing_timeline": [],
            "largest_documents": [],
            "document_details": []
        }
        
        # Analyze documents
        for doc in processed_documents:
            file_type = doc.get("file_type", "unknown")
            if file_type in report["file_types"]:
                report["file_types"][file_type] += 1
            else:
                report["file_types"][file_type] = 1
            
            # Timeline entry
            report["processing_timeline"].append({
                "filename": doc.get("filename", "Unknown"),
                "processed_at": doc.get("processed_at", ""),
                "size": doc.get("size", 0)
            })
            
            # Document details
            doc_detail = {
                "filename": doc.get("filename", "Unknown"),
                "file_type": file_type,
                "size": doc.get("size", 0),
                "chunks": doc.get("num_chunks", 0),
                "processed_at": doc.get("processed_at", "")
            }
            
            # Add file-specific metadata
            metadata = doc.get("metadata", {})
            if "pages" in metadata:
                doc_detail["pages"] = metadata["pages"]
            elif "total_sheets" in metadata:
                doc_detail["sheets"] = metadata["total_sheets"]
            
            report["document_details"].append(doc_detail)
        
        # Sort and get largest documents
        report["largest_documents"] = sorted(
            report["document_details"], 
            key=lambda x: x["size"], 
            reverse=True
        )[:10]
        
        # Sort timeline by processing date
        report["processing_timeline"].sort(key=lambda x: x["processed_at"])
        
        return report
    
    def search_conversation_history(self, chat_history: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
        """Search through conversation history"""
        if not search_term or not chat_history:
            return []
        
        search_term_lower = search_term.lower()
        matching_conversations = []
        
        for i, chat in enumerate(chat_history):
            # Search in question and answer
            question = chat.get("question", "").lower()
            answer = chat.get("answer", "").lower()
            
            if search_term_lower in question or search_term_lower in answer:
                matching_conversations.append({
                    "index": i,
                    "chat": chat,
                    "match_type": "question" if search_term_lower in question else "answer"
                })
        
        return matching_conversations
    
    def generate_source_reliability_score(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate reliability scores for sources"""
        if not sources:
            return {"error": "No sources available"}
        
        reliability_analysis = {
            "overall_score": 0,
            "source_scores": [],
            "factors": {
                "confidence_distribution": [],
                "source_diversity": 0,
                "coverage_completeness": 0
            }
        }
        
        # Calculate individual source scores
        for source in sources:
            score = source.get("score", 0)
            filename = source.get("filename", "Unknown")
            text_length = len(source.get("text", ""))
            
            # Reliability factors
            confidence_factor = score  # Direct confidence from vector similarity
            length_factor = min(text_length / 1000, 1.0)  # Longer texts might be more comprehensive
            
            source_reliability = (confidence_factor * 0.8) + (length_factor * 0.2)
            
            reliability_analysis["source_scores"].append({
                "filename": filename,
                "confidence": score,
                "reliability": source_reliability,
                "text_length": text_length
            })
            
            reliability_analysis["factors"]["confidence_distribution"].append(score)
        
        # Calculate overall metrics
        scores = [s["reliability"] for s in reliability_analysis["source_scores"]]
        reliability_analysis["overall_score"] = sum(scores) / len(scores) if scores else 0
        
        # Source diversity (unique files)
        unique_files = set(s["filename"] for s in reliability_analysis["source_scores"])
        reliability_analysis["factors"]["source_diversity"] = len(unique_files) / len(sources)
        
        # Coverage completeness (distribution of scores)
        conf_scores = reliability_analysis["factors"]["confidence_distribution"]
        if conf_scores:
            reliability_analysis["factors"]["coverage_completeness"] = 1 - (max(conf_scores) - min(conf_scores))
        
        return reliability_analysis
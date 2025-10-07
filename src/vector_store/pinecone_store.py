"""
Vector store implementation using Pinecone for document embeddings
"""
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    try:
        # Fallback for older versions
        import pinecone
        from pinecone import ServerlessSpec
        Pinecone = None
    except ImportError:
        pinecone = None
        Pinecone = None
        ServerlessSpec = None
import time
import json

logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector storage and retrieval using Pinecone"""
    
    @staticmethod
    def test_api_key(api_key: str):
        """Test if API key is valid without creating a full connection"""
        if Pinecone is None and pinecone is None:
            raise Exception("Pinecone package not installed")
        
        if not api_key or not api_key.strip():
            raise Exception("API key cannot be empty")
        
        try:
            if Pinecone:
                pc = Pinecone(api_key=api_key.strip())
                indexes = list(pc.list_indexes())
                return {"success": True, "indexes": [idx.name for idx in indexes]}
            else:
                # Legacy client test
                import pinecone
                pinecone.init(api_key=api_key.strip(), environment="us-east-1")
                indexes = pinecone.list_indexes()
                return {"success": True, "indexes": indexes}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def __init__(self, api_key: str, environment: str, index_name: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(f'sentence-transformers/{embedding_model}')
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Pinecone
        self.pc = None
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index"""
        if Pinecone is None and pinecone is None:
            raise Exception("Pinecone package not installed. Please install with: pip install pinecone")
        
        if not self.api_key:
            raise Exception("Pinecone API key is required. Please set your API key in Settings.")
        
        if not self.api_key.strip():
            raise Exception("Pinecone API key cannot be empty. Please provide a valid API key.")
        
        try:
            if Pinecone:
                # Use new Pinecone client
                logger.info("Initializing Pinecone with new client...")
                logger.info(f"API key length: {len(self.api_key)}")
                logger.info(f"API key starts with: {self.api_key[:15]}...")
                logger.info(f"API key ends with: ...{self.api_key[-10:]}")
                
                # Test API key first with a simple call
                try:
                    self.pc = Pinecone(api_key=self.api_key)
                    # Test authentication by listing indexes
                    existing_indexes = [index.name for index in self.pc.list_indexes()]
                    logger.info(f"Authentication successful. Existing indexes: {existing_indexes}")
                except Exception as auth_error:
                    logger.error(f"Authentication failed: {str(auth_error)}")
                    error_msg = str(auth_error).lower()
                    if "401" in error_msg or "unauthorized" in error_msg or "invalid" in error_msg:
                        raise Exception("Invalid Pinecone API key. Please check your API key and try again.")
                    else:
                        raise Exception(f"Pinecone authentication error: {str(auth_error)}")
                
                if self.index_name not in existing_indexes:
                    logger.info(f"Creating new Pinecone index: {self.index_name}")
                    try:
                        if ServerlessSpec:
                            # Map environment to proper cloud/region format
                            if self.environment == "us-east-1":
                                cloud = "aws"
                                region = "us-east-1"
                            elif self.environment == "us-west-2":
                                cloud = "aws" 
                                region = "us-west-2"
                            elif self.environment == "eu-west-1":
                                cloud = "aws"
                                region = "eu-west-1"
                            elif "gcp" in self.environment:
                                cloud = "gcp"
                                region = self.environment.replace("-gcp", "")
                            else:
                                # Default fallback
                                cloud = "aws"
                                region = "us-east-1"
                            
                            logger.info(f"Creating index with cloud={cloud}, region={region}")
                            self.pc.create_index(
                                name=self.index_name,
                                dimension=self.embedding_dimension,
                                metric='cosine',
                                spec=ServerlessSpec(
                                    cloud=cloud,
                                    region=region
                                )
                            )
                        else:
                            # Fallback for older API
                            self.pc.create_index(
                                name=self.index_name,
                                dimension=self.embedding_dimension,
                                metric='cosine'
                            )
                        
                        # Wait for index to be ready with timeout
                        max_wait_time = 300  # 5 minutes
                        wait_time = 0
                        while wait_time < max_wait_time:
                            index_status = self.pc.describe_index(self.index_name)
                            if index_status.status['ready']:
                                break
                            time.sleep(5)
                            wait_time += 5
                            logger.info(f"Waiting for index to be ready... ({wait_time}s)")
                        
                        if wait_time >= max_wait_time:
                            raise Exception(f"Index creation timed out after {max_wait_time} seconds")
                        
                    except Exception as create_error:
                        logger.error(f"Error creating index: {str(create_error)}")
                        raise Exception(f"Failed to create Pinecone index: {str(create_error)}")
                
                self.index = self.pc.Index(self.index_name)
                
                # Test the connection
                try:
                    stats = self.index.describe_index_stats()
                    logger.info(f"Successfully connected to Pinecone index: {self.index_name}")
                    logger.info(f"Index stats: {stats}")
                except Exception as test_error:
                    logger.error(f"Index connection test failed: {str(test_error)}")
                    # Try to provide more specific error information
                    error_str = str(test_error).lower()
                    if "401" in error_str or "unauthorized" in error_str:
                        raise Exception("Invalid Pinecone API key. Please verify your API key is correct.")
                    elif "404" in error_str or "not found" in error_str:
                        raise Exception(f"Index '{self.index_name}' not found. Please check the index name.")
                    else:
                        raise Exception(f"Failed to connect to index: {str(test_error)}")
                
            else:
                # Use legacy pinecone client
                logger.info("Initializing Pinecone with legacy client...")
                pinecone.init(api_key=self.api_key, environment=self.environment)
                
                existing_indexes = pinecone.list_indexes()
                logger.info(f"Existing indexes: {existing_indexes}")
                
                if self.index_name not in existing_indexes:
                    logger.info(f"Creating new Pinecone index: {self.index_name}")
                    try:
                        pinecone.create_index(
                            name=self.index_name,
                            dimension=self.embedding_dimension,
                            metric='cosine'
                        )
                        
                        # Wait for index to be ready
                        max_wait_time = 300
                        wait_time = 0
                        while wait_time < max_wait_time:
                            if pinecone.describe_index(self.index_name)['status']['ready']:
                                break
                            time.sleep(5)
                            wait_time += 5
                            logger.info(f"Waiting for index to be ready... ({wait_time}s)")
                        
                        if wait_time >= max_wait_time:
                            raise Exception(f"Index creation timed out after {max_wait_time} seconds")
                            
                    except Exception as create_error:
                        logger.error(f"Error creating legacy index: {str(create_error)}")
                        raise Exception(f"Failed to create Pinecone index: {str(create_error)}")
                
                self.index = pinecone.Index(self.index_name)
                
                # Test the connection
                try:
                    self.index.describe_index_stats()
                    logger.info(f"Successfully connected to Pinecone index: {self.index_name}")
                except Exception as test_error:
                    logger.error(f"Legacy index connection test failed: {str(test_error)}")
                    raise Exception(f"Failed to connect to index: {str(test_error)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            # More specific error messages
            error_msg = str(e).lower()
            if "api key" in error_msg or "unauthorized" in error_msg:
                raise Exception("Invalid Pinecone API key. Please check your API key and try again.")
            elif "environment" in error_msg:
                raise Exception(f"Invalid Pinecone environment '{self.environment}'. Please check your environment setting.")
            elif "quota" in error_msg or "limit" in error_msg:
                raise Exception("Pinecone quota exceeded. Please check your Pinecone account limits.")
            elif "network" in error_msg or "timeout" in error_msg:
                raise Exception("Network error connecting to Pinecone. Please check your internet connection.")
            else:
                raise Exception(f"Pinecone initialization failed: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    def upsert_documents(self, document_data: Dict[str, Any]) -> bool:
        """Store document chunks in Pinecone"""
        try:
            chunks = document_data.get('chunks', [])
            if not chunks:
                logger.warning("No chunks found in document data")
                return False
            
            # Prepare vectors for upsert
            vectors = []
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.generate_embeddings(texts)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{document_data['hash']}_{i}"
                
                metadata = {
                    'filename': document_data['filename'],
                    'file_type': document_data['file_type'],
                    'chunk_id': chunk['chunk_id'],
                    'start_pos': chunk['start'],
                    'end_pos': chunk['end'],
                    'text': chunk['text'][:1000],  # Truncate for metadata
                    'doc_hash': document_data['hash'],
                    'processed_at': document_data['processed_at']
                }
                
                # Add file-specific metadata
                if 'pages' in document_data['metadata']:
                    metadata['total_pages'] = document_data['metadata']['pages']
                elif 'sheets' in document_data['metadata']:
                    metadata['total_sheets'] = document_data['metadata']['total_sheets']
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert in batches to avoid limits
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully stored {len(vectors)} chunks for document: {document_data['filename']}")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting documents: {str(e)}")
            raise Exception(f"Document storage failed: {str(e)}")
    
    def search_similar(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Search in Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            results = []
            for match in search_results['matches']:
                result = {
                    'id': match['id'],
                    'score': float(match['score']),
                    'text': match['metadata'].get('text', ''),
                    'filename': match['metadata'].get('filename', ''),
                    'file_type': match['metadata'].get('file_type', ''),
                    'chunk_id': match['metadata'].get('chunk_id', 0),
                    'start_pos': match['metadata'].get('start_pos', 0),
                    'end_pos': match['metadata'].get('end_pos', 0),
                    'metadata': match['metadata']
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise Exception(f"Document search failed: {str(e)}")
    
    def delete_document(self, doc_hash: str) -> bool:
        """Delete all chunks of a document"""
        try:
            # Query to find all vectors with this doc_hash
            filter_dict = {"doc_hash": {"$eq": doc_hash}}
            
            # Delete vectors
            delete_response = self.index.delete(filter=filter_dict)
            
            logger.info(f"Deleted document with hash: {doc_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0.0)
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all stored documents (metadata only)"""
        try:
            # This is a simplified approach - in production, you might want to maintain
            # a separate metadata store for better performance
            
            # Query with empty vector to get all documents
            query_results = self.index.query(
                vector=[0.0] * self.embedding_dimension,
                top_k=10000,  # Adjust based on your needs
                include_metadata=True
            )
            
            # Group by document hash to get unique documents
            documents = {}
            for match in query_results['matches']:
                doc_hash = match['metadata'].get('doc_hash')
                if doc_hash and doc_hash not in documents:
                    documents[doc_hash] = {
                        'filename': match['metadata'].get('filename'),
                        'file_type': match['metadata'].get('file_type'),
                        'doc_hash': doc_hash,
                        'processed_at': match['metadata'].get('processed_at')
                    }
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
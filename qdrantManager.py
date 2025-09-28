# qdrant_manager.py
import asyncio
import json
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import Data.Process_data as process
import tempfile
import os
from querry_engine import QueryEngine
class QdrantDataManager:
    def __init__(self):
        self.client = QdrantClient(
            url="https://e7ccdcf9-79a9-4693-8990-8dbd4464de96.europe-west3-0.gcp.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.5hLPP46Qw0Cda9nX4cV8mEMxF97Vv3MsWAp_hCnel0I"
        )
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.collection_name = "news_collection"
        self.querry=QueryEngine()
    def add_new_article(self, text: str = None, source: str = "user", tags: List[str] = None, 
                    json_file=None, json_content: str = None) -> Dict[str, Any]:
        """Add a new article to the collection from text, file, or JSON content"""
        try:
            articles_to_add = []
            
            # Case 1: JSON file upload
            if json_file is not None:
                articles_to_add.extend(self._process_json_file(json_file))
            
            # Case 2: JSON content string
            elif json_content:
                articles_to_add.extend(self._process_json_content(json_content))
            
            # Case 3: Direct text input (original functionality)
            elif text:
                articles_to_add.append({
                    'text': text,
                    'source': source,
                    'tags': tags or []
                })
            else:
                return {"status": "error", "message": "No input provided"}
            
            # Get next available ID
            next_id = self._get_next_available_id()
            
            points = []
            results = []
            
            for i, article in enumerate(articles_to_add):
                article_text = article['text']
                article_source = article.get('source', source)
                article_tags = article.get('tags', tags or [])
                
                if not article_text.strip():
                    continue
                
                # Generate embedding
                tag_embeddings = self.model.encode(article_tags, convert_to_numpy=True)
                embedding = np.mean(tag_embeddings, axis=0).tolist()
                
                # Create point
                point = PointStruct(
                    id=next_id + i,
                    vector=embedding,
                    payload={
                        'combined_text': article_text,
                        'tags_split': article_tags,
                        'source': article_source,
                        'pubDate': pd.Timestamp.now().isoformat()
                    }
                )
                points.append(point)
                results.append({
                    "id": next_id + i,
                    "text_preview": article_text[:100] + '...' if len(article_text) > 100 else article_text,
                    "source": article_source,
                    "tags": article_tags
                })
            
            if points:
                # Insert all points
                self.client.upsert(collection_name=self.collection_name, points=points)
                
                return {
                    "status": "success", 
                    "added_count": len(points),
                    "first_id": next_id,
                    "last_id": next_id + len(points) - 1,
                    "articles": results,
                    "message": f"Successfully added {len(points)} article(s)"
                }
            else:
                return {"status": "error", "message": "No valid articles to add"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _process_json_file(self, json_file) -> List[Dict[str, Any]]:
        """Process uploaded JSON file"""
        articles = []
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
            tmp_file.write(json_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            with open(tmp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            articles.extend(self._extract_articles_from_data(data))
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
        
        return articles
    
    def _process_json_content(self, json_content: str) -> List[Dict[str, Any]]:
        """Process JSON content string"""
        try:
            data = json.loads(json_content)
            return self._extract_articles_from_data(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {e}")
    
    def _extract_articles_from_data(self, data) -> List[Dict[str, Any]]:
        """Extract articles from various JSON structures"""
        articles = []
        
        if isinstance(data, list):
            # Direct list of articles
            for item in data:
                if isinstance(item, dict):
                    articles.append(self._parse_article_dict(item))
        
        elif isinstance(data, dict):
            # Look for common keys that might contain articles
            possible_keys = ['articles', 'documents', 'news', 'data', 'items']
            for key in possible_keys:
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict):
                            articles.append(self._parse_article_dict(item))
                    break
            else:
                # If no specific key found, treat the entire dict as an article
                articles.append(self._parse_article_dict(data))
        
        return articles
    
    def _parse_article_dict(self, article_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Parse individual article dictionary with flexible field mapping"""
        # Map common field names to standard format
        text_fields = ['combined_text', 'content', 'text', 'article', 'body', 'description']
        source_fields = ['source', 'publisher', 'origin', 'website']
        tag_fields = ['tags_split', 'tags', 'categories', 'keywords']
        
        text = ''
        for field in text_fields:
            if field in article_dict and article_dict[field]:
                text = article_dict[field]
                break
        
        source = 'unknown'
        for field in source_fields:
            if field in article_dict and article_dict[field]:
                source = article_dict[field]
                break
        
        tags = []
        for field in tag_fields:
            if field in article_dict:
                if isinstance(article_dict[field], list):
                    tags = article_dict[field]
                elif isinstance(article_dict[field], str):
                    tags = [tag.strip() for tag in article_dict[field].split(',')]
                break
        
        return {
            'text': str(text) if text else '',
            'source': str(source),
            'tags': tags
        }
    
    def _get_next_available_id(self) -> int:
        """Get the next available ID in the collection"""
        try:
            # Get the highest existing ID
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                offset=0,
                with_payload=False
            )
            if points:
                return max(point.id for point in points) + 1
            else:
                return 1
        except:
            return 1
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "status": "success",
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vectors_config": str(info.config.params.vectors),
                "exists": True
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "exists": False}
    
    def search_similar_news(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar news articles"""
        try:
            query_embedding=asyncio.run(
                    self.querry.query_with_semantic_search(
                        query
                    )
                )
            # Generate embedding for query
       
            
            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for i, hit in enumerate(results):
                formatted_results.append({
                    "rank": i + 1,
                    "id": hit.id,
                    "score": hit.score,
                    "title": hit.payload.get('combined_text', '')[:100] + '...',
                    "full_text": hit.payload.get('combined_text', ''),
                    "source": hit.payload.get('source', 'unknown'),
                    "pubDate": hit.payload.get('pubDate', ''),
                    "tags": hit.payload.get('tags_split', [])
                })
            
            return formatted_results
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_all_news(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Get all news articles with pagination"""
        try:
            # Scroll through collection
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True
            )
            
            formatted_news = []
            for point in points:
                formatted_news.append({
                    "id": point.id,
                    "text_preview": point.payload.get('combined_text', '')[:200] + '...',
                    "full_text": point.payload.get('combined_text', ''),
                    "source": point.payload.get('source', 'unknown'),
                    "pubDate": point.payload.get('pubDate', ''),
                    "tags": point.payload.get('tags_split', []),
                    "text_length": len(point.payload.get('combined_text', ''))
                })
            
            return {
                "news": formatted_news,
                "total_retrieved": len(formatted_news),
                "next_offset": next_offset,
                "has_more": next_offset is not None
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_news_stats(self) -> Dict[str, Any]:
        """Get statistics about the news collection"""
        try:
            # Get all points
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your dataset size
                with_payload=True
            )
            
            sources = []
            tags = []
            text_lengths = []
            
            for point in points:
                payload = point.payload
                sources.append(payload.get('source', 'unknown'))
                tags.extend(payload.get('tags_split', []))
                text_lengths.append(len(payload.get('combined_text', '')))
            
            # Calculate statistics
            source_counts = pd.Series(sources).value_counts().to_dict()
            tag_counts = pd.Series(tags).value_counts().head(20).to_dict()  # Top 20 tags
            
            return {
                "total_articles": len(points),
                "sources_distribution": source_counts,
                "top_tags": tag_counts,
                "avg_text_length": np.mean(text_lengths) if text_lengths else 0,
                "min_text_length": min(text_lengths) if text_lengths else 0,
                "max_text_length": max(text_lengths) if text_lengths else 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def filter_news_by_source(self, source: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Filter news by source"""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                ),
                limit=limit,
                with_payload=True
            )[0]
            
            return [{
                "id": point.id,
                "text": point.payload.get('combined_text', ''),
                "source": point.payload.get('source', ''),
                "pubDate": point.payload.get('pubDate', ''),
                "tags": point.payload.get('tags_split', [])
            } for point in results]
        except Exception as e:
            return [{"error": str(e)}]
    
    def add_new_article(self, text: str, source: str = "user", tags: List[str] = None) -> Dict[str, Any]:
        """Add a new article to the collection"""
        try:
            if tags is None:
                tags = []
            
            # Generate embedding
            tag_embeddings = self.model.encode(tags, convert_to_numpy=True)
            embedding = np.mean(tag_embeddings, axis=0).tolist()
            
            # Get next ID
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                offset=0
            )
            next_id = max([point.id for point in points]) + 1 if points else 1
            
            # Create point
            point = PointStruct(
                id=next_id,
                vector=embedding,
                payload={
                    'combined_text': text,
                    'tags_split': tags,
                    'source': source,
                    'pubDate': pd.Timestamp.now().isoformat()
                }
            )
            
            # Insert
            self.client.upsert(collection_name=self.collection_name, points=[point])
            
            return {"status": "success", "id": next_id, "message": "Article added successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
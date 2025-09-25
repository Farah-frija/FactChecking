from typing import Any, Dict, List, Tuple, Optional
from pydantic import Field, BaseModel
from datetime import date
from sklearn.metrics.pairwise import cosine_similarity
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pyvis.network import Network
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import spacy
import os
import asyncio
import Data.Process_data as process
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.graphs.graph_document import Node, Relationship

from spacy.cli import download
from generate_knowledge_graph import extract_graph_data,get_tag_embeddings,get_sentence_embeddings
# Load the .env file
load_dotenv()

# Define date formats
date_formats = [
    "2025-05-13 15:33:02+00:00",
    "2024-12-25 10:30:45+00:00", 
    "2023-07-04 08:15:20+00:00",
    "2022-01-01 00:00:00+00:00",
    "2021-11-11 11:11:11+00:00"
]

# Initialize LLM and graph transformer
llm = ChatOllama(temperature=0, model="mistral")
relationship_properties = {
    "date": Field(
        ...,
        description=f"date: format YYYY-MM-DD HH:MM:SS+00:00. Format examples are {date_formats}"
    )
}

graph_transformer = LLMGraphTransformer(llm=llm, relationship_properties=relationship_properties)
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', token='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')


# Define Pydantic model for concept extraction
class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="List of concepts")

class QueryEngine():
    concept_cache = {}
    
    def __init__(self):
        # Load spaCy model for named entity recognition
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy English model...")
            
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def _extract_concepts_and_entities(self, content: str, llm: Any) -> List[str]:
        """
        Extracts concepts and named entities from the content using spaCy and a large language model.
        
        Args:
            content (str): The content from which to extract concepts and entities.
            llm: An instance of a large language model.
            
        Returns:
            list: A list of extracted concepts and entities.
        """
        if content in self.concept_cache:
            return self.concept_cache[content]
        
        # Extract named entities using spaCy
        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents]
        
        # Extract general concepts using LLM
        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts (excluding named entities) from the following text. Return only the most important 3-5 concepts:\n\n{text}\n\nKey concepts:"
        )
        
        try:
            concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)
            general_concepts = concept_chain.invoke({"text": content}).concepts_list
        except Exception as e:
            print(f"Error extracting concepts with LLM: {e}")
            # Fallback: simple keyword extraction
            general_concepts = content.split()[:5]  # Simple fallback
        
        # Combine named entities and general concepts
        all_concepts = list(set(named_entities + general_concepts))
        
        self.concept_cache[content] = all_concepts
        return all_concepts
    
    async def extract_query_concepts(self, query: str) -> List[str]:
        """
        Extract concepts and entities from a query string.
        
        Args:
            query: The query string
            
        Returns:
            List of extracted concepts and entities
        """
        return self._extract_concepts_and_entities(query, llm)
    
    async def query_with_concept_extraction(self, query: str, enhanced_graph_documents: List[Dict], 
                                         top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search using concepts extracted from the query.
        
        Args:
            query: Simple string query
            enhanced_graph_documents: List of enhanced graph documents
            top_k: Number of similar documents to return
            
        Returns:
            List of similar documents with similarity scores
        """
        # Extract concepts from query
        query_concepts = await self.extract_query_concepts(query)
        print(f"Extracted concepts from query: {query_concepts}")
        
        # Use the concepts for tag-based search
        return await self.query_with_semantic_search(
            query, enhanced_graph_documents, top_k, use_tags=True, query_concepts=query_concepts
        )
    
    async def query_with_semantic_search(self, query: str, enhanced_graph_documents: List[Dict], 
                                       top_k: int = 5, use_tags: bool = True, 
                                       query_concepts: Optional[List[str]] = None) -> List[Dict]:
        """
        Perform semantic search on the enhanced graph documents using a query string.
        
        Args:
            query: Simple string query
            enhanced_graph_documents: List of enhanced graph documents
            top_k: Number of similar documents to return
            use_tags: Whether to use tag embeddings or text embeddings
            query_concepts: Pre-extracted concepts (optional)
            
        Returns:
            List of similar documents with similarity scores
        """
        # Get query embedding
        if use_tags:
            if query_concepts is None:
                query_concepts = await self.extract_query_concepts(query)
            
            query_tags = [query_concepts]  # Use extracted concepts as tags
            query_embedding = await get_tag_embeddings(query_tags)
            query_embedding = query_embedding[0]
            doc_embeddings = np.array([doc['embedding_tag'] for doc in enhanced_graph_documents])
        else:
            query_embedding = await get_sentence_embeddings([query])
            query_embedding = query_embedding[0]
            doc_embeddings = np.array([doc['embedding'] for doc in enhanced_graph_documents])
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Get top k similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            result = {
                'document': enhanced_graph_documents[idx],
                'similarity_score': similarities[idx],
                'news_id': enhanced_graph_documents[idx]['news_id'],
                'text': enhanced_graph_documents[idx]['metadata']['original_data']['combined_text'][:],
                'tags': enhanced_graph_documents[idx]['metadata']['tags_split'],
                'source': enhanced_graph_documents[idx]['metadata']['source'],
                'pubDate':enhanced_graph_documents[idx]['metadata']['pubDate']
            }
            results.append(result)
        
        return results,query_tags


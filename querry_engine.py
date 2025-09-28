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
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from spacy.cli import download
from generate_knowledge_graph import extract_graph_data,get_tag_embeddings,get_sentence_embeddings
# Load the .env file
load_dotenv()





# Define Pydantic model for concept extraction
class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="List of concepts")

class QueryEngine():
    concept_cache = {}
    llm=ChatOllama(temperature=0, model="mistral")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', token='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')

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
        return self._extract_concepts_and_entities(query, self.llm)
    
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
    
    async def query_with_semantic_search(self, query: str,use_tags: bool = True, 
                                       query_concepts: Optional[List[str]] = None) :
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
      
        if query_concepts is None:
            query_concepts = await self.extract_query_concepts(query)
        
        query_tags = [query_concepts]  # Use extracted concepts as tags
        query_embedding = await get_tag_embeddings(query_tags)
        query_embedding = query_embedding[0]
        return query_embedding




    def analyze_fake_news_sequential(self, query: str, sorted_articles: List[Dict], max_articles: int = 5) -> Dict[str, Any]:
        """
        Analyze news articles sequentially until definitive fake/not fake verdict is reached.

        Args:
            query: The original search query
            sorted_articles: News articles from formatted_results, sorted by score (most relevant first)
            max_articles: Maximum number of articles to analyze before stopping

        Returns:
            Structured analysis with final verdict
        """
        individual_judgments = []
        definitive_verdict = None
        definitive_article_index = -1

        # Loop through articles until definitive verdict or max articles reached
        for i, article in enumerate(sorted_articles[:max_articles]):
            try:
                # Get article content using EXACT field names from formatted_results
                article_content = article.get('full_text', '')  # From your formatted_results
                article_source = article.get('source', 'unknown')  # From your formatted_results
                article_title = article.get('title', '')  # From your formatted_results
                article_score = article.get('score', 0)  # Relevance score from formatted_results
                article_rank = article.get('rank', i + 1)  # Rank from formatted_results

                print(f"Analyzing article {i+1} (Rank: {article_rank}, Score: {article_score:.3f}) from {article_source}")

                # Judge individual article against the query
                judgment = self._judge_article_against_query(query, article_content, article_source, article_score)
                print('judgmentttttttttttttt',judgment.keys())
                individual_judgments.append({
                    "article_index": i,
                    "rank": article_rank,  # Original rank from search results
                    "id": article.get('id', ''),  # Article ID
                    "source": article_source,
                    "title": article_title,
                    "score": article_score,  # Relevance score
                    "judgment": judgment,
                    "definitive": judgment['definitive']
                })

                # Check if this judgment is definitive
                if judgment['definitive']:
                    definitive_verdict = judgment
                    definitive_article_index = i
                    print(f"Definitive verdict reached at article {i+1}: {'FAKE' if judgment['is_fake'] else 'NOT FAKE'}")
                    break

            except Exception as e:
                print(f"Error analyzing article {i+1}: {e}")
                individual_judgments.append({
                    "article_index": i,
                    "rank": article.get('rank', i + 1),
                    "id": article.get('id', ''),
                    "source": article.get('source', 'unknown'),
                    "title": "Error processing article",
                    "score": article.get('score', 0),
                    "error": str(e),
                    "definitive": False
                })

        # Determine final verdict based on sequential analysis
        final_result = self._determine_final_verdict(
            query, 
            individual_judgments, 
            definitive_verdict,
            definitive_article_index,
            len(sorted_articles)
        )

        return final_result

    def _judge_article_against_query(self, query: str, article_content: str, article_source: str, article_score: float = 0) -> dict:
     """
     Judge whether a single article confirms or contradicts the query as potential fake news.
     Returns a dictionary with the judgment results.
     """
     from pydantic import BaseModel, Field

     # Define the verdict model inside the function
     class FakeNewsVerdict(BaseModel):
         is_fake: bool = Field(description="True if fake news, False if not fake")
         confidence: float = Field(description="Confidence level from 0.0 to 1.0")
         reasoning: str = Field(description="Explanation for the verdict")
         definitive: bool = Field(description="Whether this is a definitive verdict")

     # Create prompt for definitive judgment
     judgment_prompt = PromptTemplate(
         input_variables=["query", "article_content", "article_source", "article_score"],
         template="""Analyze the following news article in relation to the query. Determine if the article provides 
         definitive evidence that the query represents FAKE NEWS or NOT FAKE NEWS.

         QUERY: {query}
         ARTICLE SOURCE: {article_source}
         RELEVANCE SCORE: {article_score:.3f} (higher = more relevant to query)
         ARTICLE CONTENT: {article_content}

         Your task:
         1. If the article clearly DEBUNKS or CONTRADICTS the query as false information, answer: YES (fake)
         2. If the article clearly CONFIRMS or SUPPORTS the query as factual, answer: NO (not fake)  
         3. If the article is irrelevant, ambiguous, or doesn't provide clear evidence, answer: UNCERTAIN

         Provide a definitive YES/NO/UNCERTAIN answer when possible. Only be uncertain if truly unclear.

         Return your response in this exact JSON format:
         {{
             "is_fake": true/false,
             "confidence": 0.0-1.0,
             "reasoning": "explanation here",
             "definitive": true/false
         }}

         Answer:"""
     )

     try:
         # Use LLM to get structured judgment
         judgment_chain = judgment_prompt | self.llm.with_structured_output(FakeNewsVerdict)
         judgment = judgment_chain.invoke({
             "query": query,
             "article_content": article_content[:2000],  # Limit content length
             "article_source": article_source,
             "article_score": article_score
         })
         print('llm output',judgment)
         print('judgment',type(judgment))
         # Convert Pydantic model to dictionary
         return {
             "is_fake": judgment.is_fake,
             "confidence": judgment.confidence,
             "reasoning": judgment.reasoning,
             "definitive": judgment.definitive
         }

     except Exception as e:
         print(f"Error in individual judgment: {e}")
         # Return uncertain judgment on error
         return {
             "is_fake": False,
             "confidence": 0.3,
             "reasoning": f"Error in analysis: {str(e)}",
             "definitive": False
         }
    def _determine_final_verdict(self, query: str, individual_judgments: List[Dict], 
                               definitive_verdict: Optional[Dict], definitive_article_index: int, 
                               total_articles: int) -> Dict[str, Any]:
        """
        Determine the final verdict based on sequential analysis of articles.
        """
        if definitive_verdict and definitive_article_index >= 0:
            # We found a definitive verdict during sequential analysis
            return {
                "verdict": "fake" if definitive_verdict.get('is_fake', False) else "not fake",
                "confidence": definitive_verdict.get('confidence', 0.5),
                "articles_analyzed": definitive_article_index + 1,
                "definitive_verdict_article": definitive_article_index,
                "reasoning": f"Definitive verdict reached at article {definitive_article_index + 1}: {definitive_verdict.get('reasoning', 'No reasoning provided')}",
                "individual_judgments": individual_judgments
            }

        # No definitive verdict found - analyze patterns in judgments
        total_judgments = len(individual_judgments)

        # Count votes for fake vs not fake
        fake_votes = 0
        not_fake_votes = 0
        fake_confidences = []
        not_fake_confidences = []

        for judgment in individual_judgments:
            if 'judgment' in judgment and 'error' not in judgment:
                j = judgment['judgment']
                if j.get('is_fake', False):
                    fake_votes += 1
                    fake_confidences.append(j.get('confidence', 0))
                else:
                    not_fake_votes += 1
                    not_fake_confidences.append(j.get('confidence', 0))

        # Calculate average confidence for each side
        fake_confidence = np.mean(fake_confidences) if fake_confidences else 0
        not_fake_confidence = np.mean(not_fake_confidences) if not_fake_confidences else 0

        # Determine verdict based on majority and confidence
        if fake_votes > not_fake_votes :
            verdict = "fake"
            overall_confidence = fake_confidence
        elif not_fake_votes > fake_votes :
            verdict = "not fake" 
            overall_confidence = not_fake_confidence
        else:
            verdict = "uncertain"
            overall_confidence = max(fake_confidence, not_fake_confidence) * 0.7  # Penalty for uncertainty

        reasoning = self._generate_final_reasoning(query, individual_judgments, verdict, 
                                                 fake_votes, not_fake_votes, total_judgments)

        return {
            "verdict": verdict,
            "confidence": overall_confidence,
            "articles_analyzed": total_judgments,
            "definitive_verdict_article": -1,  # No definitive verdict
            "reasoning": reasoning,
            "individual_judgments": individual_judgments
        }

    def _generate_final_reasoning(self, query: str, judgments: List[Dict], verdict: str,
                                fake_votes: int, not_fake_votes: int, total: int) -> str:
        """Generate detailed reasoning for the final verdict."""

        if verdict == "uncertain":
            return f"""After analyzing {total} articles, no definitive consensus was reached. 
    {fake_votes} articles suggested the query might be fake, while {not_fake_votes} suggested it might be factual.
    The evidence was inconclusive, and more sources would be needed for a definitive judgment about: "{query}"."""

        elif verdict == "fake":
            return f"""Majority of analyzed articles ({fake_votes} out of {total}) indicate the query represents fake news. 
    Multiple sources contradict or debunk the claim, showing consistent patterns of misinformation about: "{query}"."""

        else:  # not fake
            return f"""Majority of analyzed articles ({not_fake_votes} out of {total}) support the query as factual. 
    The information appears consistent across multiple sources, indicating reliability for: "{query}"."""
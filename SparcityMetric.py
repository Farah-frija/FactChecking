import numpy as np
import math
from typing import Union

class SparsityContradictionDetector:
    def __init__(self, metric_type="l422", threshold=0.5):
        """
        Pure sparsity-based contradiction detector
        
        Args:
            metric_type: "l1l2", "l2l1", or "l422"
            threshold: Similarity threshold (lower = more sensitive to contradictions)
        """
        self.metric_type = metric_type
        self.threshold = threshold

    def calculate_sparsity_matrix(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Calculate sparsity-based similarity between vectors using matrix operations
        
        Args:
            vec1: Query embedding(s) of shape (1, embedding_dim) or (n_queries, embedding_dim)
            vec2: Document embeddings of shape (n_documents, embedding_dim)
            
        Returns:
            Array of contradiction scores of shape (n_queries, n_documents)
        """
        # Ensure vec1 is 2D and vec2 is 2D
        vec1 = np.atleast_2d(vec1)
        vec2 = np.atleast_2d(vec2)
        
        # Normalize vectors along the embedding dimension
        vec1_norm = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)
        
        # Calculate difference matrix using broadcasting
        # vec1_norm: (n_queries, 1, embedding_dim)
        # vec2_norm: (1, n_documents, embedding_dim)
        # diff: (n_queries, n_documents, embedding_dim)
        diff = vec1_norm[:, np.newaxis, :] - vec2_norm[np.newaxis, :, :]
        
        if self.metric_type == "l1l2":
            return self._l1l2_sparsity_matrix(diff)
        elif self.metric_type == "l2l1":
            return self._l2l1_sparsity_matrix(diff)
        elif self.metric_type == "l422":
            return self._l422_sparsity_matrix(diff)
        else:
            raise ValueError("Unknown metric type")
    
    def _l1l2_sparsity_matrix(self, diff: np.ndarray) -> np.ndarray:
        """L1/L2 ratio sparsity for matrix input"""
        # diff shape: (n_queries, n_documents, embedding_dim)
        l1_norm = np.linalg.norm(diff, ord=1, axis=2)  # shape: (n_queries, n_documents)
        l2_norm = np.linalg.norm(diff, ord=2, axis=2)  # shape: (n_queries, n_documents)
        
        # Avoid division by zero
        l2_norm_safe = np.where(l2_norm < 1e-8, 1.0, l2_norm)
        
        n = diff.shape[2]  # embedding dimension
        hoyer = (math.sqrt(n) - l1_norm / l2_norm_safe) / (math.sqrt(n) - 1)
        
        # Handle identical vectors (l2_norm ≈ 0)
        hoyer = np.where(l2_norm < 1e-8, 1.0, hoyer)
        
        return hoyer
    
    def _l2l1_sparsity_matrix(self, diff: np.ndarray) -> np.ndarray:
        """L2/L1 ratio sparsity for matrix input"""
        l1_norm = np.linalg.norm(diff, ord=1, axis=2)  # shape: (n_queries, n_documents)
        l2_norm = np.linalg.norm(diff, ord=2, axis=2)  # shape: (n_queries, n_documents)
        
        # Avoid division by zero
        l1_norm_safe = np.where(l1_norm < 1e-8, 1.0, l1_norm)
        
        ratio = l2_norm / l1_norm_safe
        
        # Handle very sparse differences (l1_norm ≈ 0)
        ratio = np.where(l1_norm < 1e-8, 0.0, ratio)
        
        return ratio
    
    def _l422_sparsity_matrix(self, diff: np.ndarray) -> np.ndarray:
        """CORRECTED L4/L2 ratio sparsity"""
        # diff shape: (n_queries, n_documents, embedding_dim)
        
        # Calculate proper norms
        l4_norm = np.sum(diff**4, axis=2) ** (1/4)  # L4 norm
        l2_norm = np.sum(diff**2, axis=2) ** (1/2)  # L2 norm
        
        # Avoid division by zero
        l2_norm_safe = np.where(l2_norm < 1e-8, 1.0, l2_norm)
        
        # Calculate L4/L2 ratio
        ratio = l4_norm / (l2_norm_safe + 1e-8)
        
        # SPARSITY INTERPRETATION:
        # - When differences are SPARSE (concentrated in few dimensions): 
        #   L4 ≈ L2 → ratio ≈ 1 → HIGH SPARSITY
        # - When differences are DENSE (spread evenly): 
        #   L4 << L2 → ratio ≈ 0 → LOW SPARSITY
        
        sparsity = ratio  # Higher ratio = more sparse
        
        # Handle identical vectors (diff ≈ 0)
        sparsity = np.where(l2_norm < 1e-8, 0.0, sparsity)  # Identical = not sparse
        
        return sparsity
    
    def detect_contradiction(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """
        Detect contradiction between embeddings using pure sparsity metrics
        
        Args:
            emb1: Query embedding(s) of shape (1, embedding_dim) or (n_queries, embedding_dim)
            emb2: Document embeddings of shape (n_documents, embedding_dim)
            
        Returns:
            Array of contradiction scores of shape (n_queries, n_documents)
            Higher values = more contradictory
        """
        # Calculate sparsity similarity matrix
        similarity_matrix = self.calculate_sparsity_matrix(emb1, emb2)
        
        # Invert to get contradiction score (higher = more contradictory)
        # Add small epsilon to avoid division by zero
        contradiction_matrix = (similarity_matrix + 1e-8)
        
        return contradiction_matrix
    
    # Keep the original single-pair method for backward compatibility
    def detect_contradiction_single(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Single pair version for backward compatibility"""
        contradiction_matrix = self.detect_contradiction(
            emb1.reshape(1, -1), 
            emb2.reshape(1, -1)
        )
        return float(contradiction_matrix[0, 0])
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import json
from google import genai
from google.genai import types
import re
import concurrent.futures

class CandidateRanker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the candidate ranker with a Sentence Transformer model.
        
        Args:
            model_name (str): The name of the Sentence Transformer model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.gemini_client = None
        
        # Initialize Gemini client if API key is available
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key)

    def generate_embeddings(self, texts):
        """
        Generate sentence embeddings for a list of texts.
        
        Args:
            texts (list): List of text strings to embed
            
        Returns:
            numpy.ndarray: Array of sentence embeddings
        """
        # Generate embeddings 
        return self.model.encode(texts, convert_to_tensor=False)
    
    def calculate_similarity(self, job_embedding, candidate_embeddings):
        """
        Calculate cosine similarity between job description and candidate resumes.
        
        Args:
            job_embedding (numpy.ndarray): Embedding of the job description
            candidate_embeddings (numpy.ndarray): Embeddings of candidate resumes
            
        Returns:
            numpy.ndarray: Array of similarity scores
        """
      
        job_embedding = job_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = util.cos_sim(job_embedding, candidate_embeddings)
        
        # Return flattened array of similarities
        return similarities.flatten()
    
    def generate_ai_summary(self, job_description, candidate_resume, similarity_score):
        """
        Generate an AI summary explaining why a candidate is a good fit.
        
        Args:
            job_description (str): The job description
            candidate_resume (str): The candidate's resume
            similarity_score (float): The calculated similarity score
            
        Returns:
            str: AI-generated summary or None if Gemini is not available
        """
        if not self.gemini_client:
            return None
        
        try:
            prompt = f"""
            Analyze why this candidate is a good fit for the job based on their resume and the job description.
            
            Job Description:
            {job_description[:10000]}
            
            Candidate Resume:
            {candidate_resume[:10000]}
            
            Similarity Score: {similarity_score:.3f}
            
            Please provide a concise 2 sentence summary explaining for each of the following points:
            1. Key matching skills/experience
            2. What makes them a strong candidate
            3. Any potential areas for growth
            4. What he lacks compared to the job description ?
            
            Keep the response professional and specific to the candidate's qualifications.
            """
            
            response = self.gemini_client.models.generate_content(
                # model= "gemma-3n" for low latency
                model="gemma-3n-e2b-it",

                contents=prompt
            )
            
            return response.text.strip() if response.text else None
            
        except Exception as e:
            print(f"Error generating AI summary: {e}")
            return None
    
    def rank_candidates(self, job_description, candidates, top_k=10, include_ai_summary=True):
        """
        Rank candidates based on their similarity to the job description.
        
        Args:
            job_description (str): The job description text
            candidates (list): List of candidate dictionaries with 'name', 'content', and 'source'
            top_k (int): Number of top candidates to return
            include_ai_summary (bool): Whether to generate AI summaries
            
        Returns:
            list: List of ranked candidates with similarity scores and optional AI summaries
        """
        if not candidates:
            return []
        
        # Prepare texts for embedding
        texts = [job_description] + [candidate['content'] for candidate in candidates]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Separate job embedding from candidate embeddings
        job_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        # Calculate similarities
        similarities = self.calculate_similarity(job_embedding, candidate_embeddings)
        
        # Create results with similarity scores
        results = []
        for i, candidate in enumerate(candidates):
            results.append({
                'name': candidate['name'],
                'content': candidate['content'],
                'source': candidate['source'],
                'similarity_score': float(similarities[i])
            })
            
        # Generate AI summaries concurrently if requested
        if include_ai_summary and self.gemini_client:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Map each future to its corresponding result dictionary
                future_to_result = {
                    executor.submit(
                        self.generate_ai_summary,
                        job_description,
                        result['content'],
                        result['similarity_score']
                    ): result for result in results
                }
                
                # As each future completes, update the result with the AI summary
                for future in concurrent.futures.as_completed(future_to_result):
                    result_dict = future_to_result[future]
                    try:
                        # CORRECTED LINE: Use .result() instead of .get()
                        ai_summary = future.result()
                        if ai_summary:
                            result_dict['ai_summary'] = ai_summary
                    except Exception as e:
                        print(f"AI summary generation failed for {result_dict['name']}: {e}")

        # Sort by similarity score in descending order
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top k results
        return results[:top_k]
    
    def get_model_info(self):
        """
        Get information about the current vectorizer.
        
        Returns:
            dict: Vectorizer information
        """
        return {
            'vectorizer_type': 'SentenceTransformer',
            'model_name': self.model._first_module().model.name_or_path
        }
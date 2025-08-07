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
import time

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
        """
        return self.model.encode(texts, convert_to_tensor=False)
    
    def calculate_similarity(self, job_embedding, candidate_embeddings):
        """
        Calculate cosine similarity between job description and candidate resumes.
        """
        job_embedding = job_embedding.reshape(1, -1)
        similarities = util.cos_sim(job_embedding, candidate_embeddings)
        return similarities.flatten()
    
    def generate_ai_summary(self, job_description, candidate_resume, similarity_score, aggressive_retry=False):
        """
        Generate an AI summary with optional aggressive retry for top candidates.
        """
        if not self.gemini_client:
            return None
        
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

        # NEW: Added aggressive retry logic for top candidates
        if aggressive_retry:
            while True: # Loop indefinitely until success
                try:
                    response = self.gemini_client.models.generate_content(
                        model="gemma-3n-e2b-it", contents=prompt
                    )
                    # Return on success, or if API gives a valid empty response
                    return response.text.strip() if response.text else None
                except Exception as e:
                    print(f"Aggressive retry for top candidate failed: {e}. Retrying in 5 seconds...")
                    time.sleep(5) # Wait longer between retries to avoid spamming
        else:
            # Standard limited retry for other cases
            for i in range(3):
                try:
                    response = self.gemini_client.models.generate_content(
                        model="gemma-3n-e2b-it", contents=prompt
                    )
                    return response.text.strip() if response.text else None
                except Exception as e:
                    print(f"AI summary generation failed on attempt {i+1}/3: {e}")
                    if i < 2:
                        time.sleep(1)
            return None

    def rank_candidates(self, job_description, candidates, top_k=10, include_ai_summary=True):
        """
        Rank candidates and generate summaries for the top K results.
        """
        if not candidates:
            return []
        
        texts = [job_description] + [c['content'] for c in candidates]
        embeddings = self.generate_embeddings(texts)
        job_embedding, candidate_embeddings = embeddings[0], embeddings[1:]
        similarities = self.calculate_similarity(job_embedding, candidate_embeddings)
        
        results = []
        for i, candidate in enumerate(candidates):
            results.append({
                'name': candidate['name'],
                'content': candidate['content'],
                'source': candidate['source'],
                'similarity_score': float(similarities[i])
            })
        
        # Sort by similarity score FIRST to identify top candidates
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Only generate summaries for the top_k candidates that will be displayed
        top_results = results[:top_k]
        
        if include_ai_summary and self.gemini_client:
            with concurrent.futures.ThreadPoolExecutor(max_workers=top_k) as executor:
                # Submit tasks only for top candidates with aggressive_retry enabled
                future_to_result = {
                    executor.submit(
                        self.generate_ai_summary,
                        job_description,
                        result['content'],
                        result['similarity_score'],
                        aggressive_retry=True # Enable indefinite retry
                    ): result for result in top_results
                }
                
                for future in concurrent.futures.as_completed(future_to_result):
                    result_dict = future_to_result[future]
                    try:
                        ai_summary = future.result()
                        if ai_summary:
                            result_dict['ai_summary'] = ai_summary
                    except Exception as e:
                        print(f"Error retrieving summary for {result_dict['name']}: {e}")


        # Return the top k results, which now include the generated summaries
        return top_results
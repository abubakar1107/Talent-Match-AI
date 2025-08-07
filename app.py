# from dotenv import load_dotenv
# load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
from candidate_ranker import CandidateRanker
from file_processor import FileProcessor
import os

# Initialize session state
if 'ranker' not in st.session_state:
    st.session_state.ranker = CandidateRanker()
if 'file_processor' not in st.session_state:
    st.session_state.file_processor = FileProcessor()

def main():
    st.title("Job Candidate Ranking System")
    st.markdown("Find the best candidates for your job using AI-powered similarity matching")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Number of top candidates to show", min_value=1, max_value=20, value=3)
        include_ai_summary = st.checkbox("Include AI-generated fit summaries", value=True)
        
        if include_ai_summary:
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            if not gemini_key:
                st.warning("!! WARNING: GEMINI_API_KEY not found in environment variables. AI summaries will be disabled.")
                include_ai_summary = False
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Job Description")
        job_description = st.text_area(
            "Enter the job description:",
            height=200,
            placeholder="Paste the job description here including requirements, responsibilities, and qualifications..."
        )
    
    with col2:
        st.header("Candidate Resumes")
        
        # File upload option
        uploaded_files = st.file_uploader(
            "Upload resume files (PDF or TXT):",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple resume files to analyze"
        )
        
        # Text input option
        st.markdown("**OR**")
        manual_resumes = st.text_area(
            "Paste resumes (separate multiple resumes with '---'):",
            height=150,
            placeholder="Resume 1: John Doe\nExperience in...\n\n---\n\nResume 2: Jane Smith\nSkilled in..."
        )
    
    # Process candidates button
    if st.button("Analyze Candidates", type="primary"):
        if not job_description.strip():
            st.error("Please enter a job description")
            return
            
        # Collect all candidate data
        candidates = []
        
        # Process uploaded files
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                for file in uploaded_files:
                    try:
                        content = st.session_state.file_processor.extract_text(file)
                        if content.strip():
                            candidates.append({
                                'name': file.name,
                                'content': content,
                                'source': 'file'
                            })
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
        
        # Process manual text input
        if manual_resumes.strip():
            manual_parts = manual_resumes.split('---')
            for i, resume in enumerate(manual_parts):
                resume = resume.strip()
                if resume:
                    # Try to extract name from first line
                    lines = resume.split('\n')
                    name = f"Candidate {i+1}"
                    if lines and len(lines[0].split()) <= 4:  # Likely a name
                        name = lines[0].strip()
                    
                    candidates.append({
                        'name': name,
                        'content': resume,
                        'source': 'manual'
                    })
        
        if not candidates:
            st.error("Please upload resume files or enter resume text")
            return
        
        # Analyze candidates
        with st.spinner(f"Analyzing {len(candidates)} candidates..."):
            try:
                results = st.session_state.ranker.rank_candidates(
                    job_description=job_description,
                    candidates=candidates,
                    top_k=top_k,
                    include_ai_summary=include_ai_summary
                )
                
                # Display results
                st.header("Top Candidates")
                st.markdown(f"Showing top {len(results)} candidates ranked by relevance")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"#{i} {result['name']} - Score: {result['similarity_score']:.3f}", expanded=i <= 3):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.metric("Similarity Score", f"{result['similarity_score']:.3f}")
                            st.write(f"**Source:** {result['source'].title()}")
                        
                        with col2:
                            if include_ai_summary and result.get('ai_summary'):
                                st.markdown("**AI Analysis:**")
                                st.write(result['ai_summary'])
                            elif include_ai_summary:
                                st.warning("AI summary not available for this candidate")
                        
                        # Show resume preview
                        with st.expander("Resume Preview"):
                            st.text_area(
                                "Resume Content:",
                                value=result['content'][:1000] + "..." if len(result['content']) > 1000 else result['content'],
                                height=150,
                                disabled=True,
                                key=f"resume_{i}"
                            )
                
                # Summary statistics
                st.header("Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Candidates", len(candidates))
                with col2:
                    st.metric("Average Score", f"{np.mean([r['similarity_score'] for r in results]):.3f}")
                with col3:
                    st.metric("Highest Score", f"{max([r['similarity_score'] for r in results]):.3f}")
                with col4:
                    st.metric("Score Range", f"{max([r['similarity_score'] for r in results]) - min([r['similarity_score'] for r in results]):.3f}")
                
                # Create downloadable results
                df = pd.DataFrame([{
                    'Rank': i+1,
                    'Name': result['name'],
                    'Similarity Score': result['similarity_score'],
                    'Source': result['source'],
                    'AI Summary': result.get('ai_summary', 'N/A') if include_ai_summary else 'N/A'
                } for i, result in enumerate(results)])
                
                st.download_button(
                    label="Download Results as CSV",
                    data=df.to_csv(index=False),
                    file_name="candidate_rankings.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()

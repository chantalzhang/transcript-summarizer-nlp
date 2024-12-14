# transcript-summarizer-nlp
COMP 550 Final Project: Generating Lecture Notes from myCourses Transcripts using NLP

How to run analyze_data.py: 

- Enter '1' for single file analysis
   - Enter the path to your text file (e.g. dataset/mycourses/lec1.txt)
- Enter '2' for directory analysis
   - Enter the path to your dataset directory (e.g. dataset/mycourses/, or just '.')
- Enter 'q' to quit


Preprocessing pipelines: 
Run run_pipeline.py. 
This script combines preprocessing techniques from preprocessor.py into 5 different pipelines implemented in pipeline.py

Quantitative valuation: 
Run evaluation.py
This script measures F1, precision and recall for extracted keywords against reference keywords, generated with RAKE and KeyBert, we should manually look over them too depending on the lecture. 
This script also measures cosine similarity to evaluate structural coherence of grouped sentences within topics. 
Attempted automate evaluation process for all 5 pipelines. This needs work still. 


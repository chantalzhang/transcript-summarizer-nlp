import os
from preprocessor import PreProcessor 

if __name__ == "__main__":
    transcript_path = "dataset/mycourses/lec4.txt" 
    preprocessor = PreProcessor()
    result = preprocessor.preprocess_transcript(transcript_path)
    print("\nKeywords:")
    print(result['keywords'])

    print("\nThematic Sections:")
    for section, content in result['thematic_sections'].items():
        print(f"\n{section}:")
        print("\n".join(content))

    print("\nCleaned Text:")
    print(result['cleaned_text'])

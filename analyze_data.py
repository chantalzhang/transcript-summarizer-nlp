import os
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path

def analyze_file(filepath):
    """Analyze a single file for word and token counts."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Count words
            words = content.split()
            word_count = len(words)
            
            # Count tokens
            tokens = word_tokenize(content)
            token_count = len(tokens)
            
            return word_count, token_count
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return 0, 0

def analyze_directory(directory_path):
    """Analyze all text files in the given directory."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    total_words = 0
    total_tokens = 0
    file_count = 0
    results = []

    dir_path = Path(directory_path)

    print("\nAnalysis Results:")
    print(f"{'Filename':<30} {'Words':<10} {'Tokens':<10}")
    print("-" * 50)
    
    for filepath in sorted(dir_path.glob('**/*.txt')):
        words, tokens = analyze_file(filepath)
        
        # Store results
        results.append({
            'filename': filepath.name,
            'words': words,
            'tokens': tokens
        })
        
        print(f"{filepath.name:<30} {words:<10} {tokens:<10}")
        
        total_words += words
        total_tokens += tokens
        file_count += 1
    
    # summary stats 
    if file_count > 0:
        avg_words = total_words / file_count
        avg_tokens = total_tokens / file_count
        
        print("\nSummary Statistics:")
        print("-" * 50)
        print(f"Total Files Analyzed: {file_count}")
        print(f"Total Words: {total_words:,}")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"Average Words per File: {avg_words:.2f}")
        print(f"Average Tokens per File: {avg_tokens:.2f}")
    else:
        print("\nNo text files found in the specified directory.")

def main():
    print("Text File Analysis Tool")
    print("-" * 20)
    
    while True:
        choice = input("\nEnter '1' for single file analysis, '2' for directory analysis, or 'q' to quit: ")
        
        if choice.lower() == 'q':
            break
            
        if choice == '1':
            filepath = input("Enter the path to your text file: ")
            if os.path.exists(filepath):
                words, tokens = analyze_file(filepath)
                print("\nAnalysis Results:")
                print(f"{'Filename':<30} {'Words':<10} {'Tokens':<10}")
                print("-" * 50)
                print(f"{Path(filepath).name:<30} {words:<10} {tokens:<10}")
            else:
                print("Error: File not found. Please enter a valid path.")
                
        elif choice == '2':
            directory = input("Enter the path to your dataset directory: ")
            if os.path.exists(directory):
                analyze_directory(directory)
            else:
                print("Error: Directory not found. Please enter a valid path.")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 
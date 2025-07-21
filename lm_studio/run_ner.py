"""
LM Studio NER Processing Script
Extracts named entities from text files using LM Studio API
"""

# Standard library imports
import json
import os
import sys
from pathlib import Path
import re

# Third-party imports
from openai import OpenAI

# Initialize LM Studio client
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
# MODEL = "gemma-3-27b-it"
# MODEL = "llama-4-scout-17b-16e-instruct"
MODEL = sys.argv[1]

# Define NER tool for LM Studio
NER_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract named entities from text and categorize them",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text of the entity"
                            },
                            "label": {
                                "type": "string",
                                "enum": [
                                    "DATE", 
                                    "PERSON", 
                                    "ORG", 
                                    "PLACE", 
                                    "SHIP", 
                                    "GROUP", 
                                    "INDICATION_OF_SLAVERY", 
                                    "RELATIONSHIPS"
                                ],
                                "description": "The category of the entity"
                            }
                        },
                        "required": ["text", "label"]
                    }
                }
            },
            "required": ["entities"]
        }
    }
}

# Configuration for batching
MAX_TOKENS_PER_CHUNK = 12000  # Conservative limit to stay well under 17000
OVERLAP_TOKENS = 500  # Overlap between chunks to avoid missing entities at boundaries

def estimate_tokens(text):
    """Rough estimate of token count (1 token ≈ 4 characters for English text)"""
    return len(text) // 4

def split_text_into_chunks(text, max_tokens=MAX_TOKENS_PER_CHUNK, overlap_tokens=OVERLAP_TOKENS):
    """Split text into overlapping chunks based on token count"""
    chunks = []
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        # Calculate end position for this chunk
        end_pos = current_pos + (max_tokens * 4)  # Convert tokens to characters
        
        if end_pos >= text_length:
            # Last chunk - go to end
            chunk = text[current_pos:]
            chunks.append(chunk)
            break
        else:
            # Find a good breaking point (end of sentence or paragraph)
            # Look for sentence endings within the last 1000 characters
            search_start = max(current_pos, end_pos - 1000)
            search_text = text[search_start:end_pos]
            
            # Try to break at sentence endings
            sentence_end = search_text.rfind('. ')
            paragraph_end = search_text.rfind('\n\n')
            
            if sentence_end != -1 and sentence_end > paragraph_end:
                # Break at sentence end
                actual_end = search_start + sentence_end + 2  # Include the period and space
            elif paragraph_end != -1:
                # Break at paragraph end
                actual_end = search_start + paragraph_end + 2  # Include the newlines
            else:
                # No good breaking point, just break at word boundary
                word_boundary = search_text.rfind(' ')
                if word_boundary != -1:
                    actual_end = search_start + word_boundary + 1
                else:
                    actual_end = end_pos
            
            chunk = text[current_pos:actual_end]
            chunks.append(chunk)
            
            # Calculate next start position with overlap
            overlap_chars = overlap_tokens * 4
            current_pos = max(current_pos + 1, actual_end - overlap_chars)
    
    return chunks

def deduplicate_entities(entities_list):
    """Deduplicate entities across chunks, keeping the most complete information"""
    seen_entities = {}
    
    for entities in entities_list:
        for entity in entities:
            text = entity.get('text', '').strip()
            label = entity.get('label', '')
            
            if not text or not label:
                continue
            
            # Create a key for deduplication
            key = (text.lower(), label)
            
            if key not in seen_entities:
                seen_entities[key] = entity
            else:
                # If we already have this entity, keep the one with more complete text
                existing = seen_entities[key]
                if len(text) > len(existing.get('text', '')):
                    seen_entities[key] = entity
    
    return list(seen_entities.values())

def read_text_file(file_path):
    """Read content from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def preprocess_text(text, filename):
    """Preprocess text to make it more digestible for the API"""
    # Check if it's the problematic file
    if "920_ROS1_(unformatted)" in filename:
        print(f"Applying special preprocessing for {filename}")
        # Remove any potentially problematic characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        text = re.sub(r'[\r\n]+', ' ', text)  # Replace multiple newlines with space
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    return text

def process_text_chunk(chunk, chunk_index, total_chunks, filename):
    """Process a single text chunk through LM Studio API for NER"""
    try:
        # Add context about which chunk this is
        chunk_info = f"[Processing chunk {chunk_index + 1} of {total_chunks} from {filename}]\n\n"
        full_text = chunk_info + chunk
        
        messages = [
            {
                "role": "user",
                "content": full_text
            }
        ]
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                # tools=[NER_TOOL],
            )
        except Exception as api_error:
            print(f"API error for {filename} chunk {chunk_index + 1}: {str(api_error)}")
            if "prediction-error" in str(api_error):
                print(f"Handling prediction error for {filename} chunk {chunk_index + 1}")
                # Try with an even shorter text
                if len(full_text) > 2000:
                    print(f"Retrying with shorter text for {filename} chunk {chunk_index + 1}")
                    shorter_text = full_text[:2000]
                    messages[0]["content"] = shorter_text
                    try:
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                        )
                    except Exception as retry_error:
                        print(f"Retry failed for {filename} chunk {chunk_index + 1}: {str(retry_error)}")
                        return {"entities": []}
                else:
                    # If text is already short, we can't do much more
                    return {"entities": []}
            else:
                # For other types of API errors
                return {"entities": []}
        
        # Get content from the response
        content = response.choices[0].message.content
        
        if content:
            print(f"Received response for {filename} chunk {chunk_index + 1}")
            
            try:
                # Clean up the content before parsing - handle whitespace and newlines
                cleaned_content = content.strip()
                parsed_data = json.loads(cleaned_content)
                print(f"Successfully parsed JSON for {filename} chunk {chunk_index + 1}")
                return parsed_data
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {filename} chunk {chunk_index + 1}: {str(e)}")
                print(f"Raw content: {content[:100]}...")
                
                # Try to extract JSON using a more lenient approach
                # Match anything that looks like a JSON object with entities
                json_match = re.search(r'(\{[\s\S]*"entities"[\s\S]*\})', content)
                if json_match:
                    try:
                        json_str = json_match.group(1)
                        # Replace any problematic characters
                        json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                        # Fix potential JSON format issues (extra commas, etc.)
                        json_str = re.sub(r',\s*\}', '}', json_str)
                        json_str = re.sub(r',\s*\]', ']', json_str)
                        
                        parsed_data = json.loads(json_str)
                        print(f"Successfully parsed JSON with regex for {filename} chunk {chunk_index + 1}")
                        return parsed_data
                    except json.JSONDecodeError as e2:
                        print(f"Secondary JSON decode error for {filename} chunk {chunk_index + 1}: {str(e2)}")
                        # Create a last-resort manual parsing fallback
                        # Look for entities in a simple pattern
                        entities = []
                        entity_matches = re.finditer(r'"text"\s*:\s*"([^"]+)"\s*,\s*"label"\s*:\s*"([^"]+)"', content)
                        for match in entity_matches:
                            text, label = match.groups()
                            entities.append({"text": text, "label": label})
                        
                        if entities:
                            print(f"Extracted {len(entities)} entities manually for {filename} chunk {chunk_index + 1}")
                            return {"entities": entities}
        
        print(f"Warning: Could not extract entities from {filename} chunk {chunk_index + 1}")
        # Save the raw content to a debug file for inspection
        with open(f"{filename}_chunk_{chunk_index + 1}_debug.txt", 'w', encoding='utf-8') as f:
            f.write(content if content else "No content")
        return {"entities": []}
    
    except Exception as e:
        print(f"Error processing text chunk {chunk_index + 1} from {filename}: {str(e)}")
        return {"entities": []}

def process_text(text, filename):
    """Process text through LM Studio API for NER, handling batching for large texts"""
    try:
        # Preprocess the text first
        text = preprocess_text(text, filename)
        
        # Estimate token count
        estimated_tokens = estimate_tokens(text)
        print(f"Estimated tokens for {filename}: {estimated_tokens}")
        
        # If text is small enough, process normally
        if estimated_tokens <= MAX_TOKENS_PER_CHUNK:
            print(f"Processing {filename} as single chunk")
            return process_text_chunk(text, 0, 1, filename)
        
        # Otherwise, split into chunks
        print(f"Splitting {filename} into chunks (estimated {estimated_tokens} tokens)")
        chunks = split_text_into_chunks(text)
        print(f"Split into {len(chunks)} chunks")
        
        # Process each chunk
        all_entities = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1}/{len(chunks)} for {filename}")
            chunk_result = process_text_chunk(chunk, i, len(chunks), filename)
            if chunk_result and 'entities' in chunk_result:
                all_entities.append(chunk_result['entities'])
        
        # Deduplicate entities across chunks
        if all_entities:
            deduplicated_entities = deduplicate_entities(all_entities)
            print(f"Found {len(deduplicated_entities)} unique entities across {len(chunks)} chunks for {filename}")
            return {"entities": deduplicated_entities}
        else:
            print(f"No entities found in any chunk for {filename}")
            return {"entities": []}
    
    except Exception as e:
        print(f"Error processing text from {filename}: {str(e)}")
        return {"entities": []}

def save_results(results, output_file):
    """Save NER results to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {str(e)}")

def process_directory(input_dir, output_dir):
    """Process all text files in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all text files
    text_files = list(input_path.glob("*.txt"))
    
    if not text_files:
        print(f"No text files found in {input_dir}")
        return
    
    print(f"Found {len(text_files)} text files to process")
    
    # Process each file
    for file_path in text_files:
        print(f"Processing {file_path.name}...")
        text = read_text_file(file_path)
        
        if text:
            # Process the text
            results = process_text(text, file_path.name)
            
            # Save results
            output_file = output_path / f"{file_path.stem}_entities.json"
            save_results(results, output_file)

def main():
    """Main function to run the NER processing"""
    if len(sys.argv) < 3:
        print("Usage: python run_ner.py <model> <input_directory> <output_directory>")
        sys.exit(1)


    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    try:
        process_directory(input_dir, output_dir)
        print("Processing complete!")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()




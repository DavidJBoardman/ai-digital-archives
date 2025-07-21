"""
LM Studio NER Processing Script
Extracts named entities from text files using LM Studio API
"""

# Standard library imports
import json
import os
import sys
from pathlib import Path

# Third-party imports
from openai import OpenAI

# Initialize LM Studio client
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
# MODEL = "gemma-3-27b-it"
MODEL = "llama-3.3-70b-instruct"

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
        import re
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        text = re.sub(r'[\r\n]+', ' ', text)  # Replace multiple newlines with space
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    # General preprocessing for all files
    # Trim very large texts
    if len(text) > 100000:
        print(f"Text too large ({len(text)} chars), trimming to 100000")
        text = text[:100000]
    
    return text

def process_text(text, filename):
    """Process text through LM Studio API for NER"""
    try:
        # Preprocess the text first
        text = preprocess_text(text, filename)
        
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                # tools=[NER_TOOL],
            )
        except Exception as api_error:
            print(f"API error for {filename}: {str(api_error)}")
            if "prediction-error" in str(api_error):
                print(f"Handling prediction error for {filename}")
                # Try with an even shorter text
                if len(text) > 2000:
                    print(f"Retrying with shorter text for {filename}")
                    shorter_text = text[:2000]
                    messages[0]["content"] = shorter_text
                    try:
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                        )
                    except Exception as retry_error:
                        print(f"Retry failed for {filename}: {str(retry_error)}")
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
            print(f"Received response for {filename}")
            
            try:
                # Clean up the content before parsing - handle whitespace and newlines
                cleaned_content = content.strip()
                parsed_data = json.loads(cleaned_content)
                print(f"Successfully parsed JSON for {filename}")
                return parsed_data
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {filename}: {str(e)}")
                print(f"Raw content: {content[:100]}...")
                
                # Try to extract JSON using a more lenient approach
                import re
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
                        print(f"Successfully parsed JSON with regex for {filename}")
                        return parsed_data
                    except json.JSONDecodeError as e2:
                        print(f"Secondary JSON decode error for {filename}: {str(e2)}")
                        # Create a last-resort manual parsing fallback
                        # Look for entities in a simple pattern
                        entities = []
                        entity_matches = re.finditer(r'"text"\s*:\s*"([^"]+)"\s*,\s*"label"\s*:\s*"([^"]+)"', content)
                        for match in entity_matches:
                            text, label = match.groups()
                            entities.append({"text": text, "label": label})
                        
                        if entities:
                            print(f"Extracted {len(entities)} entities manually for {filename}")
                            return {"entities": entities}
        
        print(f"Warning: Could not extract entities from {filename}")
        # Save the raw content to a debug file for inspection
        with open(f"{filename}_debug.txt", 'w', encoding='utf-8') as f:
            f.write(content if content else "No content")
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
        print("Usage: python gemma-3-27b-it_ner.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    try:
        process_directory(input_dir, output_dir)
        print("Processing complete!")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()




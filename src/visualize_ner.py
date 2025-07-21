"""
NER Visualization Tool
Creates an HTML report to visualize named entities in text files
"""

import json
import os
import re
from pathlib import Path
import sys
import html

def read_text_file(file_path):
    """Read content from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return ""

def read_json_file(file_path):
    """Read JSON from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {str(e)}")
        return {"entities": []}

def highlight_entities(text, entities):
    """Highlight entities in the original text"""
    # Sort entities by length (longest first) to avoid nested highlighting issues
    sorted_entities = sorted(entities, key=lambda x: len(x["text"]), reverse=True)
    
    # Escape HTML characters in the text
    escaped_text = html.escape(text)
    
    # Create a map of entity types to colors
    entity_colors = {
        "DATE": "#FFD700",  # Gold
        "PERSON": "#FF6347",  # Tomato
        "ORG": "#4682B4",  # Steel blue
        "PLACE": "#32CD32",  # Lime green
        "SHIP": "#9370DB",  # Medium purple
        "GROUP": "#FF7F50",  # Coral
        "INDICATION_OF_SLAVERY": "#DC143C",  # Crimson
        "RELATIONSHIPS": "#20B2AA"  # Light sea green
    }
    
    # Replace entities with highlighted versions
    for entity in sorted_entities:
        entity_text = entity["text"]
        entity_label = entity["label"]
        color = entity_colors.get(entity_label, "#CCCCCC")
        
        # Escape regex special characters in the entity text
        escaped_entity = re.escape(entity_text)
        
        # Use regex with word boundaries to avoid partial word matches
        pattern = r'(?<!\w)' + escaped_entity + r'(?!\w)'
        
        # Create highlighted HTML
        highlight = f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;" title="{entity_label}">{entity_text}</span>'
        
        # Replace in the text
        escaped_text = re.sub(pattern, highlight, escaped_text)
    
    return escaped_text

def create_html_report(text_dir, json_dir, output_file):
    """Create an HTML report with all files"""
    text_path = Path(text_dir)
    json_path = Path(json_dir)
    
    # Get all JSON files
    json_files = list(json_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NER Visualization</title>
        <meta charset="UTF-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                display: flex;
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            .sidebar {
                width: 250px;
                padding: 20px;
                background-color: #f0f0f0;
                border-right: 1px solid #ddd;
                height: calc(100vh - 40px);
                position: fixed;
                overflow-y: auto;
            }
            .content {
                flex: 1;
                padding: 20px;
                margin-left: 290px;
            }
            .file-link {
                display: block;
                padding: 8px;
                margin-bottom: 5px;
                background-color: #e0e0e0;
                border-radius: 3px;
                text-decoration: none;
                color: #333;
            }
            .file-link:hover {
                background-color: #d0d0d0;
            }
            .legend {
                margin-top: 20px;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 3px;
            }
            .legend-item {
                display: inline-block;
                margin: 5px;
                padding: 2px 5px;
                border-radius: 3px;
            }
            .text-container {
                white-space: pre-wrap;
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-top: 15px;
                font-size: 14px;
                line-height: 1.7;
            }
            .entity-list {
                margin-top: 20px;
                border-top: 1px solid #ddd;
                padding-top: 15px;
            }
            .entity-item {
                display: inline-block;
                margin: 5px;
                padding: 5px 8px;
                border-radius: 3px;
            }
            h1 {
                border-bottom: 2px solid #ddd;
                padding-bottom: 10px;
            }
            h2 {
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="sidebar">
                <h2>Files</h2>
                <div id="file-list">
    """
    
    # Add file links to sidebar
    for json_file in sorted(json_files):
        base_name = json_file.stem.replace('_entities', '')
        html_content += f'<a href="#{base_name}" class="file-link">{base_name}</a>\n'
    
    # Add legend to sidebar
    html_content += """
                </div>
                <div class="legend">
                    <h3>Entity Types</h3>
                    <div class="legend-item" style="background-color: #FFD700;">DATE</div>
                    <div class="legend-item" style="background-color: #FF6347;">PERSON</div>
                    <div class="legend-item" style="background-color: #4682B4;">ORG</div>
                    <div class="legend-item" style="background-color: #32CD32;">PLACE</div>
                    <div class="legend-item" style="background-color: #9370DB;">SHIP</div>
                    <div class="legend-item" style="background-color: #FF7F50;">GROUP</div>
                    <div class="legend-item" style="background-color: #DC143C;">INDICATION_OF_SLAVERY</div>
                    <div class="legend-item" style="background-color: #20B2AA;">RELATIONSHIPS</div>
                </div>
            </div>
            <div class="content">
                <h1>Named Entity Recognition Visualization</h1>
    """
    
    # Process each file
    for json_file in sorted(json_files):
        base_name = json_file.stem.replace('_entities', '')
        text_file = text_path / f"{base_name}.txt"
        
        if not text_file.exists():
            # Try alternative filenames (with variations in case or formatting)
            possible_matches = list(text_path.glob(f"{base_name}*.txt"))
            if possible_matches:
                text_file = possible_matches[0]
            else:
                print(f"No matching text file found for {base_name}")
                continue
        
        # Read the files
        text_content = read_text_file(text_file)
        json_content = read_json_file(json_file)
        
        # Get entities
        entities = json_content.get("entities", [])
        
        # Highlight entities in text
        highlighted_text = highlight_entities(text_content, entities)
        
        # Create summary of entities
        entity_summary = {}
        for entity in entities:
            label = entity["label"]
            if label not in entity_summary:
                entity_summary[label] = []
            entity_summary[label].append(entity["text"])
        
        # Add file section to HTML
        html_content += f"""
                <div id="{base_name}" class="file-section">
                    <h2>{base_name}</h2>
                    <p><strong>Original file:</strong> {text_file.name}</p>
                    <p><strong>Entities found:</strong> {len(entities)}</p>
                    
                    <h3>Text with Highlighted Entities</h3>
                    <div class="text-container">
                        {highlighted_text}
                    </div>
                    
                    <h3>Entity Summary</h3>
                    <div class="entity-list">
        """
        
        # Add entity summary
        for label, items in entity_summary.items():
            color = {
                "DATE": "#FFD700",
                "PERSON": "#FF6347",
                "ORG": "#4682B4",
                "PLACE": "#32CD32",
                "SHIP": "#9370DB",
                "GROUP": "#FF7F50",
                "INDICATION_OF_SLAVERY": "#DC143C",
                "RELATIONSHIPS": "#20B2AA"
            }.get(label, "#CCCCCC")
            
            html_content += f'<h4>{label} ({len(items)})</h4>\n<div>\n'
            for item in sorted(set(items)):
                html_content += f'<span class="entity-item" style="background-color: {color};">{html.escape(item)}</span>\n'
            html_content += '</div>\n'
        
        html_content += """
                    </div>
                </div>
                <hr>
        """
    
    # Close HTML tags
    html_content += """
            </div>
        </div>
        <script>
            // Scroll to anchor on load
            window.onload = function() {
                if (window.location.hash) {
                    const element = document.querySelector(window.location.hash);
                    if (element) {
                        element.scrollIntoView();
                    }
                }
            };
        </script>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report created at {output_file}")

def main():
    """Main function to run the visualization tool"""
    if len(sys.argv) < 4:
        print("Usage: python visualize_ner.py <text_directory> <json_directory> <output_html_file>")
        sys.exit(1)
    
    text_dir = sys.argv[1]
    json_dir = sys.argv[2]
    output_file = sys.argv[3]
    
    try:
        create_html_report(text_dir, json_dir, output_file)
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
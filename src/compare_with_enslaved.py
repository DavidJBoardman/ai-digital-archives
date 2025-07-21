#!/usr/bin/env python3
"""
Person Matching Script for Digital Archives

This script compares people found in entity extraction spreadsheets with the enslaved persons database.
It calculates likelihood/probability scores for potential matches based on:
- Name similarity (exact, partial, phonetic)
- Date proximity and overlap
- Location proximity
- Contextual information

Usage:
    python wp4/compare_with_enslaved.py [spreadsheet_path] [enslaved_json_path] [--output output.csv] [--threshold 0.3]
"""

import json
import pandas as pd
import argparse
import re
import sys
import os
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
from datetime import datetime
import Levenshtein
from collections import defaultdict

class PersonMatcher:
    def __init__(self, enslaved_data_path: str):
        """Initialize the person matcher with enslaved database."""
        self.enslaved_data = self._load_enslaved_data(enslaved_data_path)
        self.enslaved_persons = self._extract_persons_from_enslaved_data()
        
        # Weights for different matching factors
        self.weights = {
            'name_exact': 1.0,
            'name_partial': 0.7,
            'name_phonetic': 0.5,
            'date_overlap': 0.8,
            'date_proximity': 0.6,
            'location_match': 0.9,
            'context_similarity': 0.4
        }
        
    def _load_enslaved_data(self, path: str) -> List[Dict]:
        """Load the enslaved database JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading enslaved data: {e}")
            return []
    
    def _extract_persons_from_enslaved_data(self) -> List[Dict]:
        """Extract person entries from the enslaved database."""
        persons = []
        name_pattern = re.compile(r'^[A-Z][a-zA-Z\s\'-]+[a-zA-Z]$')
        
        for item in self.enslaved_data:
            if not isinstance(item, dict):
                continue
                
            # Check if this looks like a person
            labels = item.get('labels', {})
            if 'en' not in labels:
                continue
                
            name = labels['en'].get('value', '')
            
            # Filter for actual person names (not categories or concepts)
            if (len(name.split()) >= 2 and 
                name_pattern.match(name) and 
                not any(word in name.lower() for word in [
                    'vocabulary', 'service', 'labor', 'trade', 'person', 
                    'community', 'document', 'occupation', 'type'
                ])):
                
                person_data = self._extract_person_metadata(item)
                if person_data:
                    persons.append(person_data)
        
        print(f"Extracted {len(persons)} persons from enslaved database")
        return persons
    
    def _extract_person_metadata(self, item: Dict) -> Optional[Dict]:
        """Extract relevant metadata from a person record."""
        try:
            person = {
                'id': item.get('id'),
                'name': item.get('labels', {}).get('en', {}).get('value', ''),
                'description': item.get('descriptions', {}).get('en', {}).get('value', ''),
                'aliases': [alias.get('value', '') for alias in item.get('aliases', {}).get('en', [])],
                'dates': [],
                'locations': [],
                'occupations': [],
                'relationships': [],
                'sources': []
            }
            
            # Extract additional data from claims
            claims = item.get('claims', {})
            for prop, values in claims.items():
                for value in values:
                    self._process_claim(person, prop, value)
            
            return person
        except Exception as e:
            print(f"Error processing person record: {e}")
            return None
    
    def _process_claim(self, person: Dict, property_id: str, claim: Dict):
        """Process individual claims to extract metadata."""
        try:
            mainsnak = claim.get('mainsnak', {})
            datavalue = mainsnak.get('datavalue', {})
            
            if datavalue.get('type') == 'string':
                value = datavalue.get('value', '')
                # Try to categorize the string value
                if self._looks_like_date(value):
                    person['dates'].append(value)
                elif self._looks_like_location(value):
                    person['locations'].append(value)
                elif self._looks_like_occupation(value):
                    person['occupations'].append(value)
            
            elif datavalue.get('type') == 'time':
                time_value = datavalue.get('value', {})
                if isinstance(time_value, dict) and 'time' in time_value:
                    person['dates'].append(time_value['time'])
            
            # Extract qualifiers for additional context
            qualifiers = claim.get('qualifiers', {})
            for qual_prop, qual_values in qualifiers.items():
                for qual_value in qual_values:
                    qual_datavalue = qual_value.get('datavalue', {})
                    if qual_datavalue.get('type') == 'string':
                        val = qual_datavalue.get('value', '')
                        if self._looks_like_location(val):
                            person['locations'].append(val)
        
        except Exception:
            pass  # Skip malformed claims
    
    def _looks_like_date(self, text: str) -> bool:
        """Check if text looks like a date."""
        date_patterns = [
            r'\d{4}',  # Year
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'[A-Za-z]+ \d{1,2}, \d{4}',  # Month DD, YYYY
            r'\d{1,2} [A-Za-z]+ \d{4}',  # DD Month YYYY
        ]
        return any(re.search(pattern, text) for pattern in date_patterns)
    
    def _looks_like_location(self, text: str) -> bool:
        """Check if text looks like a location."""
        # Simple heuristics for location detection
        location_indicators = [
            'county', 'parish', 'plantation', 'estate', 'town', 'city', 
            'village', 'island', 'river', 'port', 'district', 'province'
        ]
        return (len(text.split()) <= 4 and 
                (text[0].isupper() or 
                 any(indicator in text.lower() for indicator in location_indicators)))
    
    def _looks_like_occupation(self, text: str) -> bool:
        """Check if text looks like an occupation."""
        occupation_words = [
            'work', 'labor', 'servant', 'field', 'house', 'cook', 'driver',
            'carpenter', 'blacksmith', 'weaver', 'seamstress', 'nurse'
        ]
        return any(word in text.lower() for word in occupation_words)
    
    def calculate_name_similarity(self, name1: str, name2: str) -> Dict[str, float]:
        """Calculate various name similarity scores."""
        name1_clean = self._clean_name(name1)
        name2_clean = self._clean_name(name2)
        
        scores = {}
        
        # Exact match
        scores['exact'] = 1.0 if name1_clean.lower() == name2_clean.lower() else 0.0
        
        # Partial match (any word in common)
        words1 = set(name1_clean.lower().split())
        words2 = set(name2_clean.lower().split())
        common_words = words1.intersection(words2)
        if words1 and words2:
            scores['partial'] = len(common_words) / max(len(words1), len(words2))
        else:
            scores['partial'] = 0.0
        
        # Levenshtein distance similarity
        if name1_clean and name2_clean:
            distance = Levenshtein.distance(name1_clean.lower(), name2_clean.lower())
            max_len = max(len(name1_clean), len(name2_clean))
            scores['levenshtein'] = 1.0 - (distance / max_len) if max_len > 0 else 0.0
        else:
            scores['levenshtein'] = 0.0
        
        # Sequence matcher similarity
        scores['sequence'] = SequenceMatcher(None, name1_clean.lower(), name2_clean.lower()).ratio()
        
        return scores
    
    def _clean_name(self, name: str) -> str:
        """Clean and normalize name for comparison."""
        # Remove titles, honorifics, and common prefixes/suffixes
        title_patterns = [
            r'\b(Mr|Mrs|Miss|Ms|Dr|Rev|Sir|Lady|Lord|Captain|Col|Major|Gen)\.\s*',
            r'\b(Jr|Sr|III|IV|Esq)\b',
            r'\([^)]*\)',  # Remove content in parentheses
        ]
        
        cleaned = name
        for pattern in title_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def calculate_date_similarity(self, dates1: List[str], dates2: List[str]) -> float:
        """Calculate similarity between date ranges."""
        if not dates1 or not dates2:
            return 0.0
        
        # Extract years from date strings
        years1 = self._extract_years(dates1)
        years2 = self._extract_years(dates2)
        
        if not years1 or not years2:
            return 0.0
        
        # Calculate overlap and proximity
        min1, max1 = min(years1), max(years1)
        min2, max2 = min(years2), max(years2)
        
        # Check for overlap
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)
        
        if overlap_start <= overlap_end:
            # There's an overlap
            overlap_years = overlap_end - overlap_start + 1
            total_range = max(max1, max2) - min(min1, min2) + 1
            return overlap_years / total_range
        else:
            # No overlap, calculate proximity
            gap = overlap_start - overlap_end
            if gap <= 10:  # Within 10 years
                return max(0.0, 1.0 - (gap / 20.0))  # Linear decay over 20 years
            else:
                return 0.0
    
    def _extract_years(self, dates: List[str]) -> List[int]:
        """Extract years from date strings."""
        years = []
        for date_str in dates:
            # Look for 4-digit years
            year_matches = re.findall(r'\b(1[5-9]\d{2}|20\d{2})\b', str(date_str))
            for year_str in year_matches:
                try:
                    year = int(year_str)
                    if 1500 <= year <= 2100:  # Reasonable year range
                        years.append(year)
                except ValueError:
                    continue
        return years
    
    def calculate_location_similarity(self, locations1: List[str], locations2: List[str]) -> float:
        """Calculate similarity between location lists."""
        if not locations1 or not locations2:
            return 0.0
        
        # Normalize locations
        norm_loc1 = {self._normalize_location(loc) for loc in locations1}
        norm_loc2 = {self._normalize_location(loc) for loc in locations2}
        
        # Calculate Jaccard similarity
        intersection = norm_loc1.intersection(norm_loc2)
        union = norm_loc1.union(norm_loc2)
        
        if union:
            return len(intersection) / len(union)
        else:
            return 0.0
    
    def _normalize_location(self, location: str) -> str:
        """Normalize location names for comparison."""
        # Remove common suffixes and normalize
        normalized = re.sub(r'\b(County|Parish|Plantation|Estate|Town|City)\b', '', location, flags=re.IGNORECASE)
        normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
        return normalized
    
    def calculate_context_similarity(self, person1: Dict, person2: Dict) -> float:
        """Calculate similarity based on context (occupations, descriptions, etc.)."""
        score = 0.0
        factors = 0
        
        # Compare occupations
        if person1.get('occupations') and person2.get('occupations'):
            occ_sim = self.calculate_location_similarity(person1['occupations'], person2['occupations'])
            score += occ_sim
            factors += 1
        
        # Compare descriptions (simple keyword overlap)
        desc1 = person1.get('description', '').lower()
        desc2 = person2.get('description', '').lower()
        
        if desc1 and desc2:
            words1 = set(re.findall(r'\w+', desc1))
            words2 = set(re.findall(r'\w+', desc2))
            if words1 and words2:
                desc_sim = len(words1.intersection(words2)) / len(words1.union(words2))
                score += desc_sim
                factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def calculate_match_probability(self, spreadsheet_person: Dict, enslaved_person: Dict) -> Dict:
        """Calculate overall match probability between two persons."""
        
        # Name similarity
        name_scores = self.calculate_name_similarity(
            spreadsheet_person.get('name', ''), 
            enslaved_person.get('name', '')
        )
        
        # Date similarity
        date_score = self.calculate_date_similarity(
            spreadsheet_person.get('dates', []),
            enslaved_person.get('dates', [])
        )
        
        # Location similarity
        location_score = self.calculate_location_similarity(
            spreadsheet_person.get('locations', []),
            enslaved_person.get('locations', [])
        )
        
        # Context similarity
        context_score = self.calculate_context_similarity(spreadsheet_person, enslaved_person)
        
        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0
        
        # Use best name score
        best_name_score = max(name_scores.values())
        if best_name_score > 0:
            if name_scores['exact'] > 0:
                total_score += name_scores['exact'] * self.weights['name_exact']
                total_weight += self.weights['name_exact']
            elif name_scores['partial'] > 0.5:
                total_score += name_scores['partial'] * self.weights['name_partial']
                total_weight += self.weights['name_partial']
            else:
                total_score += best_name_score * self.weights['name_phonetic']
                total_weight += self.weights['name_phonetic']
        
        if date_score > 0:
            if date_score > 0.7:
                total_score += date_score * self.weights['date_overlap']
                total_weight += self.weights['date_overlap']
            else:
                total_score += date_score * self.weights['date_proximity']
                total_weight += self.weights['date_proximity']
        
        if location_score > 0:
            total_score += location_score * self.weights['location_match']
            total_weight += self.weights['location_match']
        
        if context_score > 0:
            total_score += context_score * self.weights['context_similarity']
            total_weight += self.weights['context_similarity']
        
        # Calculate final probability
        probability = total_score / total_weight if total_weight > 0 else 0.0
        
        return {
            'probability': probability,
            'name_scores': name_scores,
            'date_score': date_score,
            'location_score': location_score,
            'context_score': context_score,
            'enslaved_person': enslaved_person
        }
    
    def find_matches(self, spreadsheet_persons: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """Find potential matches for persons from spreadsheet in enslaved database."""
        matches = []
        
        for sp_person in spreadsheet_persons:
            person_matches = []
            
            for enslaved_person in self.enslaved_persons:
                match_result = self.calculate_match_probability(sp_person, enslaved_person)
                
                if match_result['probability'] >= threshold:
                    person_matches.append(match_result)
            
            # Sort by probability (descending)
            person_matches.sort(key=lambda x: x['probability'], reverse=True)
            
            # Add top matches
            for match in person_matches[:5]:  # Top 5 matches
                matches.append({
                    'spreadsheet_person': sp_person,
                    'match_probability': match['probability'],
                    'enslaved_person_id': match['enslaved_person']['id'],
                    'enslaved_person_name': match['enslaved_person']['name'],
                    'name_exact': match['name_scores']['exact'],
                    'name_partial': match['name_scores']['partial'],
                    'name_levenshtein': match['name_scores']['levenshtein'],
                    'name_sequence': match['name_scores']['sequence'],
                    'date_score': match['date_score'],
                    'location_score': match['location_score'],
                    'context_score': match['context_score'],
                    'enslaved_description': match['enslaved_person']['description'],
                    'enslaved_dates': ', '.join(map(str, match['enslaved_person']['dates'])),
                    'enslaved_locations': ', '.join(match['enslaved_person']['locations']),
                    'enslaved_aliases': ', '.join(match['enslaved_person']['aliases'])
                })
        
        return matches


def load_spreadsheet_data(file_path: str) -> List[Dict]:
    """Load and process person data from the spreadsheet."""
    try:
        df = pd.read_excel(file_path)
        
        persons = []
        current_person = None
        
        for _, row in df.iterrows():
            # Group data by document and extract persons
            document = row.get('Document', '')
            person_name = row.get('PERSON', '')
            date_info = row.get('DATE', '')
            location_info = row.get('PLACE', '')
            
            if pd.notna(person_name) and person_name.strip():
                # Create person record
                person_data = {
                    'name': person_name.strip(),
                    'document': document,
                    'dates': [date_info] if pd.notna(date_info) and date_info.strip() else [],
                    'locations': [location_info] if pd.notna(location_info) and location_info.strip() else [],
                    'description': f"Found in document: {document}",
                    'occupations': [],
                    'row_data': row.to_dict()
                }
                persons.append(person_data)
        
        print(f"Extracted {len(persons)} persons from spreadsheet")
        return persons
        
    except Exception as e:
        print(f"Error loading spreadsheet: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Match persons from spreadsheet with enslaved database")
    parser.add_argument('spreadsheet', nargs='?', default='lm_studio/outputs/experiment_3/entity_extraction_results_gemma-3.xlsx',
                       help='Path to the entity extraction spreadsheet')
    parser.add_argument('enslaved_db', nargs='?', default='datasets/enslaved/enslaved.json',
                       help='Path to the enslaved database JSON file')
    parser.add_argument('--output', '-o', default='wp4/outputs/person_matches.csv',
                       help='Output CSV file for matches')
    parser.add_argument('--threshold', '-t', type=float, default=0.3,
                       help='Minimum probability threshold for matches (0.0-1.0)')
    parser.add_argument('--max-matches', '-m', type=int, default=5,
                       help='Maximum number of matches per person')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.spreadsheet):
        print(f"Error: Spreadsheet file not found: {args.spreadsheet}")
        sys.exit(1)
    
    if not os.path.exists(args.enslaved_db):
        print(f"Error: Enslaved database file not found: {args.enslaved_db}")
        sys.exit(1)
    
    print("Loading data...")
    
    # Load spreadsheet data
    spreadsheet_persons = load_spreadsheet_data(args.spreadsheet)
    if not spreadsheet_persons:
        print("No persons found in spreadsheet")
        sys.exit(1)
    
    # Initialize matcher
    matcher = PersonMatcher(args.enslaved_db)
    if not matcher.enslaved_persons:
        print("No persons found in enslaved database")
        sys.exit(1)
    
    print(f"Searching for matches with threshold {args.threshold}...")
    
    # Find matches
    matches = matcher.find_matches(spreadsheet_persons, threshold=args.threshold)
    
    if matches:
        # Create output dataframe
        matches_df = pd.DataFrame(matches)
        
        # Reorder columns for better readability
        column_order = [
            'spreadsheet_person', 'enslaved_person_name', 'match_probability',
            'name_exact', 'name_partial', 'name_levenshtein', 'name_sequence',
            'date_score', 'location_score', 'context_score',
            'enslaved_person_id', 'enslaved_description', 'enslaved_dates',
            'enslaved_locations', 'enslaved_aliases'
        ]
        
        # Flatten spreadsheet_person data
        matches_df['spreadsheet_name'] = matches_df['spreadsheet_person'].apply(lambda x: x.get('name', ''))
        matches_df['spreadsheet_document'] = matches_df['spreadsheet_person'].apply(lambda x: x.get('document', ''))
        matches_df['spreadsheet_dates'] = matches_df['spreadsheet_person'].apply(lambda x: ', '.join(x.get('dates', [])))
        matches_df['spreadsheet_locations'] = matches_df['spreadsheet_person'].apply(lambda x: ', '.join(x.get('locations', [])))
        
        # Drop the complex column and reorder
        matches_df = matches_df.drop('spreadsheet_person', axis=1)
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        matches_df.to_csv(args.output, index=False)
        
        print(f"\nFound {len(matches)} potential matches")
        print(f"Results saved to: {args.output}")
        
        # Print summary statistics
        high_confidence = len(matches_df[matches_df['match_probability'] >= 0.7])
        medium_confidence = len(matches_df[(matches_df['match_probability'] >= 0.5) & (matches_df['match_probability'] < 0.7)])
        low_confidence = len(matches_df[matches_df['match_probability'] < 0.5])
        
        print(f"\nMatch confidence breakdown:")
        print(f"High confidence (≥0.7): {high_confidence}")
        print(f"Medium confidence (0.5-0.7): {medium_confidence}")  
        print(f"Low confidence (<0.5): {low_confidence}")
        
        # Show top matches
        print(f"\nTop 10 matches:")
        top_matches = matches_df.nlargest(10, 'match_probability')
        for _, match in top_matches.iterrows():
            print(f"  {match['spreadsheet_name']} → {match['enslaved_person_name']} "
                  f"(prob: {match['match_probability']:.3f})")
    
    else:
        print("No matches found above the threshold")


if __name__ == "__main__":
    main()

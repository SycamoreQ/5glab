import json 
import re
from typing import Dict , Optional , List , Any 
from datetime import datetime 
import ollama 
import openai
import spacy

class LLMQueryParser: 
    
    def __init__(self , model: str = "llama3.2", use_local: bool = True): 
        self.model = model 
        self.use_local = use_local
        self.current_year = datetime.now().year
        if self.use_local: 
            pass 
        else: 
            self.client = openai.OpenAI()
            
        self.system_prompt = self._build_system_prompt()



    def _build_system_prompt(self) -> str: 
        """Create system prompt for query parsing."""
        return f"""You are a research query parser. Extract structured information from academic search queries.

                Current year: {datetime.now().year}

                Extract these facets from the query:
                1. **semantic**: Core research topic/keywords (remove temporal/author/venue keywords)
                2. **temporal**: Year range as [start_year, end_year] or null
                - "recent" = last 4 years
                - "latest" = last 1-2 years
                - "classic"/"old" = pre-2010
                - Explicit years: use those
                3. **author**: Author name if mentioned, or null
                4. **author_search_mode**: true if query is asking for author's work, false otherwise
                5. **institutional**: University/company affiliation if mentioned, or null
                6. **venue**: Conference/journal name if mentioned, or null
                7. **intent**: One of [survey, methodology, application, theory, empirical, author_works] or null
                8. **entities**: List of key technical terms/topics

                Respond ONLY with valid JSON. No explanation.

                Examples:

                Query: "what is the recent publication by author XYZ"
                {{
                "semantic": "publications",
                "temporal": [{self.current_year - 4}, {self.current_year}],
                "author": "XYZ",
                "author_search_mode": true,
                "institutional": null,
                "venue": null,
                "intent": "author_works",
                "entities": []
                }}

                Query: "deep learning papers on medical imaging from Stanford 2020-2023"
                {{
                "semantic": "deep learning medical imaging",
                "temporal": [2020, 2023],
                "author": null,
                "author_search_mode": false,
                "institutional": "Stanford",
                "venue": null,
                "intent": "application",
                "entities": ["deep learning", "medical imaging"]
                }}

                Query: "survey on transformer architectures in NeurIPS"
                {{
                "semantic": "transformer architectures",
                "temporal": null,
                "author": null,
                "author_search_mode": false,
                "institutional": null,
                "venue": "NeurIPS",
                "intent": "survey",
                "entities": ["transformer", "architectures"]
                }}

                Query: "latest work by Geoffrey Hinton on neural networks"
                {{
                "semantic": "neural networks",
                "temporal": [{self.current_year - 2}, {self.current_year}],
                "author": "Geoffrey Hinton",
                "author_search_mode": true,
                "institutional": null,
                "venue": null,
                "intent": "author_works",
                "entities": ["neural networks"]
                }}

                Now parse the user's query."""
    
    def parse_query(self , query:str) -> Dict[str ,Any]: 
        if self.use_local: 
            return self._parse_local(query)
        else: 
            return self._parse_openai(query)
        

    def _parse_openai(self , query:str) -> Dict[str , Any]: 
        try: 
            response = self.client.chat.completions.create(
                model = self.model,
                messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature= 0.0,
                max_tokens = 300,
                response_format= {"type" : "json_object"}
            )

            result = response.choices[0].message.content
            facets = json.loads(result)

            facets['original'] = query
            facets = self._validate_and_fix(facets)
            
            return facets
        

        except Exception as e: 
            print(f"LLM parsing failed: {e}. Falling back to rule-based.")
            return self._fallback_parse(query)
        

    def _parse_local(self , query: str ) -> Dict[str , Any]: 
        try: 
            import requests 
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json ={
                    'model' : 'llama3.2',
                    'prompt' : f"{self.system_prompt}\n\nQuery: {query}\n\nJSON:",
                    'stream' : False,
                    'format' : 'json'
                }
            )

            result = response.json()['response']
            facets = json.loads(result)
            facets['original'] = query
            facets = self._validate_and_fix(facets)
            
            return facets
            
        except Exception as e : 
            print(f"Local LLM parsing failed: {e}. Falling back.")
            return self._fallback_parse(query)


    def _validate_and_fix(self, facets: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix LLM output."""
        # Ensure all required keys exist
        defaults = {
            'semantic': '',
            'temporal': None,
            'author': None,
            'author_search_mode': False,
            'institutional': None,
            'venue': None,
            'intent': None,
            'entities': [],
            'original': ''
        }
        
        for key, default in defaults.items():
            if key not in facets:
                facets[key] = default
        
        if facets['temporal'] and isinstance(facets['temporal'], list):
            if len(facets['temporal']) == 2:
                facets['temporal'] = tuple(facets['temporal'])
            else:
                facets['temporal'] = None
        
        if not isinstance(facets['entities'], list):
            facets['entities'] = []
        
        return facets
    

    def _fallback_parse(self, query: str) -> Dict[str, Any]:
        """Fallback to rule-based parsing if LLM fails."""    
        fallback_parser = EnhancedQueryParser()
        return fallback_parser.parse(query)


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None


class EnhancedQueryParser:
    """Robust parser for natural language research queries."""
    
    def __init__(self):
        self.temporal_keywords = {
            'recent': (datetime.now().year - 4, datetime.now().year),  # Last 4 years
            'latest': (datetime.now().year - 1, datetime.now().year),
            'new': (datetime.now().year - 2, datetime.now().year),
            'old': (1900, 2010),
            'classic': (1900, 2000),
            'modern': (2015, datetime.now().year),
            'contemporary': (2018, datetime.now().year),
            'last year': (datetime.now().year - 1, datetime.now().year - 1),
            'this year': (datetime.now().year, datetime.now().year)
        }
        
        self.intent_keywords = {
            'survey': ['survey', 'review', 'overview', 'summary', 'state of the art'],
            'methodology': ['method', 'approach', 'technique', 'algorithm', 'how to'],
            'application': ['application', 'use case', 'implementation', 'applied'],
            'theory': ['theory', 'theoretical', 'proof', 'analysis', 'mathematical'],
            'empirical': ['experiment', 'empirical', 'evaluation', 'benchmark', 'results']
        }
        
        self.institutions = [
            'MIT', 'Stanford', 'Berkeley', 'CMU', 'Harvard',
            'Google', 'Microsoft', 'Meta', 'DeepMind', 'OpenAI',
            'Oxford', 'Cambridge', 'ETH', 'Imperial', 'Princeton'
        ]
        
        # Common question patterns
        self.question_patterns = [
            r'what (?:is|are) (?:the )?(.*?)(?:\?|$)',
            r'(?:show|find|get|retrieve) (?:me )?(.*?)(?:\?|$)',
            r'(?:tell me about|information on|details about) (.*?)(?:\?|$)',
            r'(?:papers|work|research|publications?) (?:on|about) (.*?)(?:\?|$)'
        ]
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Enhanced parsing with better author/natural language handling.
        
        Example:
            "what is the recent publication by author XYZ"
            -> {
                'semantic': "publication",
                'temporal': (2021, 2024),
                'author': "XYZ",
                'author_search_mode': True,  # NEW: Indicates author-centric query
                'intent': 'author_works'
            }
        """
        query_lower = query.lower()
        
        facets = {
            'semantic': query,
            'temporal': None,
            'institutional': None,
            'intent': None,
            'entities': [],
            'author': None,
            'author_search_mode': False,  
            'venue': None,
            'original': query,
            'paper_title': None,
            'paper_search_mode': False,
            'paper_operation': None,
            'relation_focus': None,
            'hop_depth': None
            }
        
        #  Detect if this is an author-centric query
        if self._is_author_query(query_lower):
            facets['author_search_mode'] = True
            facets['intent'] = 'author_works'
        
        # Extract author name (improved)
        author = self._extract_author_robust(query)
        if author:
            facets['author'] = author
        
        # Extract temporal constraints
        temporal = self._extract_temporal(query_lower)
        if temporal:
            facets['temporal'] = temporal
        
        # Extract institution
        institution = self._extract_institution(query)
        if institution:
            facets['institutional'] = institution
        
        # Extract venue
        venue = self._extract_venue(query)
        if venue:
            facets['venue'] = venue
        
        # Detect intent (if not author-centric)
        if not facets['author_search_mode']:
            intent = self._detect_intent(query_lower)
            if intent:
                facets['intent'] = intent
        
        # Clean semantic query
        semantic_query = self._clean_semantic_query_robust(
            query, temporal, institution, author, venue
        )
        facets['semantic'] = semantic_query
        
        # Extract entities
        if nlp and semantic_query:
            entities = self._extract_entities(semantic_query)
            facets['entities'] = entities
        
        return facets
    
    def _is_author_query(self, query: str) -> bool:
        """Detect if query is asking about specific author's work."""
        patterns = [
            r'\bby author\b',
            r'\bby\s+[A-Z][a-z]+',  # "by Smith", "by John"
            r'\bauthor\s+[A-Z]',     # "author XYZ"
            r'\bpublications? (?:by|from|of)\b',
            r'\bpapers? (?:by|from|of)\b',
            r'\bwork (?:by|from|of)\b',
            r'\bwritten by\b',
            r'\bauthored by\b'
        ]
        
        return any(re.search(pattern, query) for pattern in patterns)
    
    def _extract_author_robust(self, query: str) -> Optional[str]:
        """
        Enhanced author extraction that handles:
        - Real names (via spaCy NER)
        - Placeholder names like "XYZ", "ABC"
        - Patterns like "author John Smith", "by Jane Doe"
        """
        # Pattern 1: Explicit "author [Name]" or "by [Name]"
        patterns = [
            r'\bauthor\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',  # "author John Smith"
            r'\bby\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',      # "by John Smith"
            r'\bfrom\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',    # "from Jane Doe"
            r'\bwritten by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
            r'\bauthored by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                author_name = match.group(1).strip()
                # Handle placeholder names like "XYZ"
                if len(author_name) >= 2:  # Accept even "XY"
                    return author_name
        
        # Pattern 2: Use spaCy NER (for real names)
        if nlp:
            doc = nlp(query)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Check if it's actually an author mention (not just any person)
                    context = query.lower()
                    if any(keyword in context for keyword in ['author', 'by', 'written', 'paper']):
                        return ent.text
        
        # Pattern 3: Capitalized word after "by" (fallback)
        match = re.search(r'\bby\s+([A-Z]+)', query)
        if match:
            return match.group(1)
        
        return None
    
    def _clean_semantic_query_robust(
        self, 
        query: str, 
        temporal, 
        institution, 
        author, 
        venue
    ) -> str:
        """Enhanced cleaning that preserves semantic content."""
        cleaned = query
        
        # Remove question words/phrases
        question_stems = [
            r'\bwhat (?:is|are) (?:the )?\b',
            r'\bshow me\b',
            r'\bfind me\b',
            r'\btell me about\b',
            r'\bget me\b',
            r'\bretrieve\b',
            r'\blooking for\b',
            r'\bi want\b',
            r'\bi need\b'
        ]
        
        for pattern in question_stems:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove temporal keywords
        for keyword in self.temporal_keywords.keys():
            cleaned = re.sub(rf'\b{keyword}\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove year mentions
        cleaned = re.sub(r'\b(19|20)\d{2}\b', '', cleaned)
        cleaned = re.sub(r'\b(19|20)\d{2}\s*-\s*(19|20)\d{2}\b', '', cleaned)
        
        # Remove author mention (but keep the name for searching!)
        if author:
            # Remove "by author XYZ" but keep "XYZ" if it's part of topic
            cleaned = re.sub(rf'\bauthor\s+{re.escape(author)}\b', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(rf'\bby\s+{re.escape(author)}\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove institution
        if institution:
            cleaned = cleaned.replace(institution, '')
        
        # Remove venue
        if venue:
            cleaned = cleaned.replace(venue, '')
            cleaned = re.sub(r'\bin\b|\bat\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove filler words
        fillers = [
            'papers?', 'publications?', 'work', 'research', 'study', 'studies',
            'from', 'about', 'on', 'regarding', 'concerning'
        ]
        for filler in fillers:
            cleaned = re.sub(rf'\b{filler}\b', '', cleaned, flags=re.IGNORECASE)
        
        # Clean whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # If nothing left, return original or a default
        if not cleaned or len(cleaned) < 3:
            # For author queries, use a generic "papers by author"
            if author:
                cleaned = f"publications by {author}"
            else:
                cleaned = query  # Fall back to original
        
        return cleaned
    
    def _extract_temporal(self, query: str) -> Optional[tuple]:
        """Same as before, but with better "recent" handling."""
        # Check keyword-based temporal
        for keyword, year_range in self.temporal_keywords.items():
            if keyword in query:
                return year_range
        
        # Check explicit years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, query)
        if years:
            years = [int(y) for y in years]
            if len(years) == 1:
                return (years[0], years[0])
            elif len(years) >= 2:
                return (min(years), max(years))
        
        # Check year ranges
        range_pattern = r'\b(19|20)\d{2}\s*-\s*(19|20)\d{2}\b'
        match = re.search(range_pattern, query)
        if match:
            start, end = match.group().split('-')
            return (int(start.strip()), int(end.strip()))
        
        return None
    
    def _extract_institution(self, query: str) -> Optional[str]:
        """Same as before."""
        for inst in self.institutions:
            if inst.lower() in query.lower():
                return inst
        
        pattern = r'(?:at|from|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        match = re.search(pattern, query)
        if match:
            return match.group(1)
        
        return None
    
    def _detect_intent(self, query: str) -> Optional[str]:
        """Same as before."""
        for intent, keywords in self.intent_keywords.items():
            if any(kw in query for kw in keywords):
                return intent
        return None
    
    def _extract_venue(self, query: str) -> Optional[str]:
        """Same as before."""
        venues = ['CVPR', 'ICCV', 'NeurIPS', 'ICML', 'ACL', 'EMNLP', 
                  'KDD', 'WWW', 'SIGIR', 'Nature', 'Science']
        
        for venue in venues:
            if venue.lower() in query.lower():
                return venue
        
        pattern = r'(?:in|at)\s+([A-Z]+(?:\s+[A-Z]+)*)'
        match = re.search(pattern, query)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_entities(self, query: str) -> List[str]:
        """Same as before."""
        if not nlp:
            return []
        
        doc = nlp(query)
        entities = []
        
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                entities.append(chunk.text)
        
        return entities


    

        
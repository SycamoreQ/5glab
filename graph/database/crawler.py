import requests
import time

API_KEY = "YOUR_API_KEY_HERE"
headers = {
    "X-ELS-APIKey": API_KEY,
    "Accept": "application/json"
}

subjects = ["COMP", "MATH", "ENGI"]  

BASE_URL = "https://api.elsevier.com/content/search/scopus"


for subj in subjects:
    start = 0
    page_size = 25 
    max_results = 100  # limit per subject 
    
    while start < max_results:
        params = {
            "query": f"SUBJAREA({subj})",
            "count": page_size,
            "start": start
        }
        
        resp = requests.get(BASE_URL, headers=headers, params=params)
        data = resp.json()
        
        entries = data.get("search-results", {}).get("entry", [])
        if not entries:
            break 
        
        for entry in entries:
            title = entry.get("dc:title")
            doi = entry.get("prism:doi")
            creator = entry.get("dc:creator")
            cover_date = entry.get("prism:coverDate")
            cit_count = entry.get("citedby-count")
            
            print(f"[{subj}] {title} | {doi} | {creator} | {cover_date}")
        
        start += page_size
        time.sleep(1)  

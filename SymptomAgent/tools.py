from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Create driver instance
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def get_diseases_from_neo4j(user_symptoms, top_n=5):
    """
    Query Neo4j to find diseases matching the given symptoms.
    
    Args:
        user_symptoms (list): List of symptom strings
        top_n (int): Maximum number of diseases to return
    
    Returns:
        list: List of dictionaries with disease information
    """
    with driver.session() as session:
        try:
            result = session.run("""
                MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
                WHERE toLower(s.name) IN $symptom_list
                WITH d, collect(s.name) as matched_symptoms, count(*) as match_count
                ORDER BY match_count DESC
                RETURN d.name as disease, matched_symptoms, match_count
                LIMIT $top_n
            """, symptom_list=[s.lower().strip() for s in user_symptoms], top_n=top_n)
            
            return [
                {
                    "disease": record["disease"],
                    "matched_symptoms": record["matched_symptoms"],
                    "match_count": record["match_count"]
                }
                for record in result
            ]
        except Exception as e:
            print(f"Error querying Neo4j: {e}")
            return []

def get_all_symptoms_from_neo4j():
    """
    Retrieve all available symptoms from Neo4j database.
    
    Returns:
        list: List of all symptom names
    """
    with driver.session() as session:
        try:
            result = session.run("MATCH (s:Symptom) RETURN s.name as symptom ORDER BY s.name")
            return [record["symptom"] for record in result]
        except Exception as e:
            print(f"Error retrieving symptoms: {e}")
            return []

def close_driver():
    """Close the Neo4j driver connection."""
    if driver:
        driver.close()
from rapidfuzz import process
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

# ---- Rule-based known typo/abbreviation corrections ----
SYMPTOM_MAP = {
    "fvr": "fever",
    "htn": "hypertension",
    "vommiting": "vomiting",
    "abd pain": "abdominal pain",
    "c/o cp": "chest pain",
    "sorethroat": "sore throat",
    "haedache": "headache",
    "sob": "shortness of breath",
    "loosemotions": "diarrhea"
}

# ---- Load ClinicalBERT tokenizer and model ----
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model.eval()

# ---- Get ClinicalBERT CLS embedding ----
def get_clinicalbert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # [CLS] token

# ---- Fuzzy string matcher ----
def fuzzy_match(symptom, all_symptoms, threshold=80):
    match, score, _ = process.extractOne(symptom, all_symptoms)
    return match if score > threshold else None

# ---- ClinicalBERT semantic similarity ----
def get_closest_symptom_with_bert(symptom, all_symptoms):
    user_vec = get_clinicalbert_embedding(symptom.lower())
    max_sim = -1
    best_match = None
    for s in all_symptoms:
        s_vec = get_clinicalbert_embedding(s.lower())
        sim = cosine_similarity(user_vec, s_vec).item()
        if sim > max_sim:
            max_sim = sim
            best_match = s
    return best_match, max_sim

# ---- Final normalization function ----
def normalize_symptom(symptom, all_symptoms):
    s = symptom.lower().strip()

    # Layer 1: Rule-based
    if s in SYMPTOM_MAP:
        return SYMPTOM_MAP[s]

    # Layer 2: Exact match
    if s in [x.lower() for x in all_symptoms]:
        return s

    # Layer 3: Fuzzy string match
    fuzzy = fuzzy_match(s, all_symptoms)
    if fuzzy:
        return fuzzy

    # Layer 4: ClinicalBERT semantic match
    semantic, score = get_closest_symptom_with_bert(s, all_symptoms)
    return semantic if score > 0.85 else None


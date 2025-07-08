from crewai import Crew
from .agents import create_symptom_checker_agent
from .task import create_diagnosis_task
import os
from dotenv import load_dotenv
from .tools import get_diseases_from_neo4j, get_all_symptoms_from_neo4j, close_driver, normalize_symptom

# Load environment variables
load_dotenv()

def main():
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            print("Error: OPENAI_API_KEY not found in environment variables.")
            return

        agent = create_symptom_checker_agent(openai_api_key)

        # Get all available symptoms from Neo4j
        print("Loading symptoms from knowledge graph...")
        all_symptoms = get_all_symptoms_from_neo4j()

        if not all_symptoms:
            print("Error: Could not retrieve symptoms from database.")
            return

        print("============================================")
        print("Available symptoms (sample):")
        print(", ".join(all_symptoms[:30]))
        if len(all_symptoms) > 30:
            print("...")
        print(f"\nTotal symptoms in database: {len(all_symptoms)}")
        print("============================================")
        print("Enter symptoms separated by comma (e.g. cough, fever, headache):")

        user_in = input("Your symptoms: ").strip()

        if not user_in:
            print("No symptoms entered. Exiting.")
            return

        user_symptoms = [s.strip() for s in user_in.split(',') if s.strip()]

        valid_symptoms = []
        invalid_symptoms = []

        for symptom in user_symptoms:
            normalized = normalize_symptom(symptom, all_symptoms)
            if normalized:
                valid_symptoms.append(normalized)
            else:
                invalid_symptoms.append(symptom)

        if invalid_symptoms:
            print(f"\nWarning: These symptoms were not recognized: {', '.join(invalid_symptoms)}")

        if not valid_symptoms:
            print("No valid symptoms found in database. Please check your input.")
            return

        print(f"\nProcessing symptoms: {', '.join(valid_symptoms)}")

        # Query Neo4j for top matching diseases
        print("\nQuerying knowledge graph for possible diseases...\n")
        matched_diseases = get_diseases_from_neo4j(valid_symptoms, top_n=5)

        if not matched_diseases:
            print("No diseases matched your symptoms in the knowledge graph.\n")
        else:
            print("Top matches from knowledge graph:")
            for idx, d in enumerate(matched_diseases, 1):
                print(f"{idx}. {d['disease']} (Matched: {', '.join(d['matched_symptoms'])} | Score: {d['match_count']})")
            print()

        print("Generating AI analysis...")
        diagnosis_task = create_diagnosis_task(agent, valid_symptoms, matched_diseases)
        crew = Crew(
            agents=[agent],
            tasks=[diagnosis_task],
            verbose=True
        )

        results = crew.kickoff()
        print("\n" + "="*50)
        print("SYMPTOM CHECKER AI ANALYSIS")
        print("="*50)
        print(results)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        close_driver()

if __name__ == "__main__":
    main()
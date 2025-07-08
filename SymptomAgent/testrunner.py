import pandas as pd
from tools import get_diseases_from_neo4j, get_all_symptoms_from_neo4j, normalize_symptom

# Load test dataset
df = pd.read_csv("reduced_disease_dataset.csv")
all_symptoms = get_all_symptoms_from_neo4j()

# Prepare test cases: (symptom list, expected disease)
test_cases = []
for _, row in df.iterrows():
    expected = row["diseases"].strip().lower()
    symptom_list = [col for col in df.columns[1:] if row[col] == 1]
    test_cases.append((symptom_list, expected))

# Evaluate each case
top1_correct = 0
top3_correct = 0
results = []

print("\n================== TEST RUNNER LOG ==================\n")

for idx, (symptoms, expected) in enumerate(test_cases):
    normalized = [normalize_symptom(s, all_symptoms) for s in symptoms if normalize_symptom(s, all_symptoms)]
    predicted = get_diseases_from_neo4j(normalized, top_n=3)
    predicted_names = [d['disease'].lower() for d in predicted]

    top1_hit = predicted_names[0] == expected if predicted_names else False
    top3_hit = expected in predicted_names

    if top1_hit:
        top1_correct += 1
    if top3_hit:
        top3_correct += 1

    print(f"🧪 Test Case #{idx+1}")
    print(f"   Input symptoms     : {', '.join(symptoms)}")
    print(f"   Normalized         : {', '.join(normalized) if normalized else 'None'}")
    print(f"   Expected disease   : {expected}")
    print(f"   Predicted diseases : {', '.join(predicted_names) if predicted_names else 'None'}")
    print(f"   Top-1 Correct?     : {'✅' if top1_hit else '❌'}")
    print(f"   Top-3 Correct?     : {'✅' if top3_hit else '❌'}")
    print("------------------------------------------------------")

    results.append({
        "Input Symptoms": ", ".join(symptoms),
        "Normalized": ", ".join(normalized),
        "Expected Disease": expected,
        "Predicted Diseases": ", ".join(predicted_names),
        "Top-1 Match": top1_hit,
        "Top-3 Match": top3_hit
    })

# Print final accuracy summary
total = len(test_cases)
print("\n================== ACCURACY SUMMARY ==================\n")
print(f"Total Test Cases  : {total}")
print(f"Top-1 Accuracy    : {top1_correct}/{total} → {top1_correct / total:.2%}")
print(f"Top-3 Accuracy    : {top3_correct}/{total} → {top3_correct / total:.2%}")
print("\n======================================================\n")

# Optionally export CSV
results_df = pd.DataFrame(results)
results_df.to_csv("agent_test_results.csv", index=False)
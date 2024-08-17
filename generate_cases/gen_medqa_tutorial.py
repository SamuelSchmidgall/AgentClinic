import json, openai, re, time
from datasets import load_dataset

# Set OpenAI key
openai.api_key = "insert-openai-api-key-here"

# Extract the testing set for the MedQA dataset
medqa_test_set = load_dataset("bigbio/med_qa")["test"]

# Extract all case studies from MedQA
case_studies = [case for case in medqa_test_set if "likely diagnosis?" in case["question"]]

# Randomize cases (optional)
import random
random.shuffle(case_studies)

# How many cases studies to generate
cases_to_gen = 108 - 78

# Provide an example of the OSCE template
examples = """
Here is an example of the structure:
{
  "OSCE_Examination": {
    "Objective_for_Doctor": "Assess and diagnose the patient presenting with acute abdominal pain.",
    "Patient_Actor": {
      "Demographics": "30-year-old female",
      "History": "The patient complains of sudden onset of sharp, right lower quadrant abdominal pain since last night. The pain has progressively worsened over the last 12 hours. She mentions that she felt nauseous this morning but has not vomited. No recent changes in bowel habits or urinary symptoms have been reported.",
      "Symptoms": {
        "Primary_Symptom": "Sharp, right lower quadrant abdominal pain",
        "Secondary_Symptoms": ["Nausea", "No vomiting", "No change in bowel habits", "No urinary symptoms"]
      },
      "Past_Medical_History": "No significant past medical history. No previous surgeries.",
      "Social_History": "Non-smoker, occasional alcohol use. Works as a software developer.",
      "Review_of_Systems": "Denies fever, vomiting, diarrhea, dysuria, or flank pain."
    },
    "Physical_Examination_Findings": {
      "Vital_Signs": {
        "Temperature": "37.2°C (99°F)",
        "Blood_Pressure": "120/75 mmHg",
        "Heart_Rate": "78 bpm",
        "Respiratory_Rate": "16 breaths/min"
      },
      "Abdominal_Examination": {
        "Inspection": "No distension or visible masses.",
        "Auscultation": "Normal bowel sounds.",
        "Percussion": "Tympanic throughout, no shifting dullness.",
        "Palpation": "Tenderness in the right lower quadrant. No guarding or rebound tenderness. Rovsing's sign positive, suggesting peritoneal irritation."
      }
    },
    "Test_Results": {
      "Complete_Blood_Count": {
        "WBC": "12,000 /μL (elevated)",
        "Hemoglobin": "13.5 g/dL",
        "Platelets": "250,000 /μL"
      },
      "Urinalysis": {
        "Appearance": "Clear",
        "WBC": "2-5 /HPF",
        "RBC": "0-2 /HPF",
        "Nitrites": "Negative",
        "Leukocyte_Esterase": "Negative"
      },
      "Imaging": {
        "Ultrasound_Abdomen": {
          "Findings": "Enlarged appendix with wall thickening and fluid collection. No evidence of ovarian cyst or ectopic pregnancy."
        }
      }
    },
    "Correct_Diagnosis": "Acute Appendicitis",
  }
}
"""

outp_str = ""
cases_generated = 0
for _case in case_studies:
    messages = [
        {"role": "system", "content": "Please generate a sample Objective Structured Clinical Examination (OSCE) for the patient actor and the doctor, including what the correct diagnosis should be as a structured json. Only provide the doctor with the objective and provide \"test results\" as a separate category. Provide these for a primary care doctor exam."},
        {"role": "user", "content": " Generate a OSCE for the following case study {}.".format(_case) + "Please read the \"answer\" category for the correct diagnosis. \n\nHere is an example of correct the OSCE format" + examples + """\n\nPlease create a new one here:\n"""}
    ]
    # Generate OSCE json
    response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=messages,
        )
    # Remove potential garbage
    answer = response["choices"][0]["message"]["content"]
    answer = re.sub("\s+", " ", answer)
    answer = answer.replace("```json ", "")
    answer = answer.replace("```", "")
    try: 
      # Make sure diagnoses match
      if _case["answer"].lower() != json.loads(answer)["OSCE_Examination"]["Correct_Diagnosis"].lower():
         continue
      
      outp_str += answer + "\n"
      # add it to the JSON
      with open("grounded.jsonl", "w") as f:
          f.write(outp_str)
      cases_generated += 1
    except Exception: 
      pass
    # Total number of cases to generate met
    if cases_generated >= cases_to_gen:
       exit()
    time.sleep(1)

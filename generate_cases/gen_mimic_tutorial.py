import os, csv
import json, openai, re, time

# Set OpenAI key
openai.api_key = "insert-openai-api-key-here"

# First install the MIMIC-IV dataset from 
# https://physionet.org/content/mimiciv/2.2/
# And place it into this folder (generated_cases)

# Change this according to need
base_str = "./"

patient_info = dict()
with open(base_str + "hosp/admissions.csv", "r") as f:
    admit_file = list(csv.reader(f))
    print("Done")
with open(base_str + "hosp/diagnoses_icd.csv", "r") as f:
    diagn_file = list(csv.reader(f))
    print("Done")
with open(base_str + "hosp/d_icd_diagnoses.csv", "r") as f:
    diag_ids_file = list(csv.reader(f))
    print("Done")
with open(base_str + "hosp/patients.csv", "r") as f:
    pats_file = list(csv.reader(f))
    print("Done")


rev_diag_code = {_df[0]:_df[2] for _df in diag_ids_file[1:]}

admit_labels = admit_file[0]
diagn_labels = diagn_file[0]

rev_admit_labels = {admit_labels[_i]:_i for _i in range(len(admit_labels))}
rev_diag_labels = {diagn_labels[_i]:_i for _i in range(len(diagn_labels))}

print(admit_labels)
print(diagn_labels)

for csv_line in range(1, len(admit_file)):
    pat_id = admit_file[csv_line][0]
    if pat_id not in patient_info:
        patient_info[pat_id] = dict() # subject id
        patient_info[pat_id]["tests"] = dict()
        patient_info[pat_id]["history"] = list()
        patient_info[pat_id]["diagnosis"] = -1 #list()
        patient_info[pat_id]["diag_imp"] = 9999 #list()
        patient_info[pat_id]["demographics"] = dict()
        patient_info[pat_id]["demographics"]["race"] = admit_file[csv_line][12]

diagnoses = {}
num_diagnoses = {}
for csv_line in range(1, len(diagn_file)):
    pat_id = diagn_file[csv_line][0]
    if pat_id not in num_diagnoses:
        num_diagnoses[pat_id] = 0
        diagnoses[pat_id] = []
    diagn = rev_diag_code[diagn_file[csv_line][3]]
    if "history" in diagn.lower():
        patient_info[pat_id]["history"].append(diagn)
    else:
        num_diagnoses[pat_id] += 1
        try:
            patient_info[pat_id]["diagnosis"] = diagn
            patient_info[pat_id]["diag_imp"] = int(diagn_file[csv_line][2])
            diagnoses[pat_id].append(diagn)
        except Exception: pass

num = 0
patlist = []
# Choose only cases with diagnoses == 1
for _ in num_diagnoses:
    if num_diagnoses[_] < 2:
        num += 1
        if num >= 300: break
        patlist.append(_)

pats_file = [_ for _ in pats_file if _[0] in patlist]

with open(base_str + "hosp/omr.csv", "r") as f:
    omr_file = list(csv.reader(f))
    print("Done")
with open(base_str + "hosp/microbiologyevents.csv", "r") as f:
    micro_file = list(csv.reader(f))
    print("Done")
with open(base_str + "hosp/labevents.csv", "r") as f:
    labenvt_file = list(csv.reader(f))
    print("Done")
with open(base_str + "hosp/d_labitems.csv", "r") as f:
    labitems_file = list(csv.reader(f))


rev_item_code = {_df[0]:_df[1] + " " + _df[2] for _df in labitems_file[1:]}
micro_labels  = micro_file[0]
labenvt_labels  = labenvt_file[0]
rev_micro_labels = {micro_labels[_i]:_i for _i in range(len(micro_labels))}
rev_labevnt_labels = {labenvt_labels[_i]:_i for _i in range(len(labenvt_labels))}


for csv_line in range(1, len(pats_file)):
    patient_info[pats_file[csv_line][0]]["demographics"]["gender"] = pats_file[csv_line][1]
    patient_info[pats_file[csv_line][0]]["demographics"]["anchor_age"] = pats_file[csv_line][2]


for csv_line in range(1, len(omr_file)):
    # take the first presenting one
    try:
        if omr_file[csv_line][3] not in patient_info[omr_file[csv_line][0]]["tests"]:
            patient_info[omr_file[csv_line][0]]["tests"][omr_file[csv_line][3]] = omr_file[csv_line][4]
    except Exception:
        pass

for csv_line in range(1, len(micro_file)):
    # take the first presenting one
    try:
        patient_info[micro_file[csv_line][1]]["tests"][micro_file[csv_line][rev_micro_labels["test_name"]].lower()] = micro_file[csv_line][rev_micro_labels["comments"]].lower()
    except Exception:
        pass

for csv_line in range(1, len(labenvt_file)):
    # ignore empty ones
    try:
        value = labenvt_file[csv_line][rev_labevnt_labels["value"]]
        test = rev_item_code[labenvt_file[csv_line][rev_labevnt_labels["itemid"]]]
        if "_" not in value and len(value) > 0 and test not in patient_info[labenvt_file[csv_line][1]]["tests"]:
            patient_info[labenvt_file[csv_line][1]]["tests"][test] = value
    except Exception:
        pass

case_studies = patient_info

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
      outp_str += answer + "\n"
      # add it to the JSON
      with open("grounded.jsonl", "w") as f:
          f.write(outp_str)
    except Exception: 
      pass
    
    time.sleep(1)

import argparse
import anthropic
import os
from transformers import pipeline
import re, random, time, json, replicate, os
from openai import OpenAI
client 
llama2_url = "meta/llama-2-70b-chat"
llama3_url = "meta/meta-llama-3-70b-instruct"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"

def load_huggingface_model(model_name):
    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    return pipe

def inference_huggingface(prompt, pipe):
    response = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
    response = response.replace(prompt, "")
    return response


def query_model(model_str, prompt, system_prompt, tries=3, timeout=15.0,
                image_requested=False, scene=None, max_prompt_len=2**14, clip_prompt=False):
    global client 
    SUPPORTED = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano",
        "gpt4", "gpt3.5", "gpt4o", "gpt-4o-mini", "gpt4v",
        "claude3.5sonnet", "o1",
        "llama-2-70b-chat", "mixtral-8x7b", "llama-3-70b-instruct"
    }
    if model_str not in SUPPORTED and "_HF" not in model_str:
        raise Exception(f"No model by the name {model_str}")

    for _ in range(tries):
        if clip_prompt:
            prompt = prompt[:max_prompt_len]
        try:
            # -------------- Vision branch --------------
            if image_requested:
                # GPT-5 (nano/mini/base) handled as text here; block vision for them explicitly
                if model_str in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
                    # GPT-5 vision via Responses API
                    resp = client.responses.create(
                        model=model_str,  # e.g., "gpt-5"
                        input=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "input_image",
                                    # Use either a URL or base64
                                    "image_url": {"url": scene.image_url}  # e.g., "https://.../image.png"
                                    # OR, for base64:
                                    # "image_data": {"b64": scene.image_b64}
                                }
                            ]
                        }],

                        max_completion_tokens=200,
                    )
                    # Responses API returns messages under output_text (or output[0].content etc. depending on SDK)
                    ans = resp.output_text
                    return re.sub(r"\s+", " ", ans)
                if model_str == "gpt4v":
                    resp = client.chat.completions.create(
                        model="gpt-4v",  # or "gpt-4o-mini"
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": scene.image_url}},
                            ]},
                        ],
                        temperature=0.05,
                        max_tokens=200,
                    )
                    ans = resp.choices[0].message.content
                    return re.sub(r"\s+", " ", ans)
                elif model_str == "gpt-4o-mini":
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",  # or "gpt-4o-mini"
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": scene.image_url}},
                            ]},
                        ],
                        temperature=0.05,
                        max_tokens=200,
                    )
                    ans = resp.choices[0].message.content
                    return re.sub(r"\s+", " ", ans)
                elif model_str == "gpt4":
                    resp = client.chat.completions.create(
                        model="gpt-4",  # or "gpt-4o-mini"
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": scene.image_url}},
                            ]},
                        ],
                        temperature=0.05,
                        max_tokens=200,
                    )
                    ans = resp.choices[0].message.content
                    return re.sub(r"\s+", " ", ans)
                elif model_str == "gpt4o":
                    resp = client.chat.completions.create(
                        model="gpt-4o",  # or "gpt-4o-mini"
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": scene.image_url}},
                            ]},
                        ],
                        temperature=0.05,
                        max_tokens=200,
                    )
                    ans = resp.choices[0].message.content
                    return re.sub(r"\s+", " ", ans)
                else:
                    raise Exception(f"Vision not supported for model: {model_str}")
                

            # -------------- Text branch --------------
            def _msg():
                return [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

            if model_str in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
                system_prompt += (
    " Respond in 1–3 sentences maximum. "
    "If brevity is critical, respond in a single sentence."
)
                resp = client.responses.create(
                    model=model_str,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt}
                        ],
                    }],
                   metadata={
        "verbosity": "low"   # <- pass inside metadata
    }
                )
                ans = None
                if hasattr(resp, "output_text") and resp.output_text:
                    ans = resp.output_text
                else:
                    try:
                        ans = resp.output[0].content[0].text
                    except Exception:
                        ans = "[NO OUTPUT]"

                return re.sub(r"\s+", " ", ans)

            if model_str == "gpt4":
                resp = client.chat.completions.create(
                model="gpt-4.1",  # use a current model id
                messages=[{"role":"system","content":system_prompt},
                        {"role":"user","content":prompt}],
                temperature=0.05, max_tokens=200
            )
                ans = resp.choices[0].message.content
                return re.sub(r"\s+", " ", ans)

            elif model_str == "gpt4v":
                resp = client.chat.completions.create(
                model="gpt-4-vision-preview",  # use a current model id
                messages=[{"role":"system","content":system_prompt},
                        {"role":"user","content":prompt}],
                temperature=0.05, max_tokens=200
            )
                ans = resp.choices[0].message.content
                return re.sub(r"\s+", " ", ans)

            elif model_str == "gpt-4o-mini":
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":system_prompt},
                            {"role":"user","content":prompt}],
                    temperature=0.05, max_tokens=200
                )
                ans = resp.choices[0].message.content
                return re.sub(r"\s+", " ", ans)

            elif model_str == "o1":
                resp = client.responses.create(
                 model="o1",
                 input=[
                  {"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}
                  ],
                )
                ans = resp.output_text
                return re.sub(r"\s+", " ", ans)

            elif model_str == "gpt3.5":
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"system","content":system_prompt},
                            {"role":"user","content":prompt}],
                    temperature=0.05, max_tokens=200
                )
                ans = resp.choices[0].message.content
                return re.sub(r"\s+", " ", ans)

            elif model_str == "claude3.5sonnet":
                # client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt, max_tokens=256,
                    messages=[{"role": "user", "content": prompt}]
                )
                ans = json.loads(message.to_json())["content"][0]["text"]
                return re.sub(r"\s+", " ", ans)

            elif model_str == "gpt4o":
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role":"system","content":system_prompt},
                            {"role":"user","content":prompt}],
                    temperature=0.05, max_tokens=200
                )
                ans = resp.choices[0].message.content
                return re.sub(r"\s+", " ", ans)
                

            elif model_str == 'llama-2-70b-chat':
                output = replicate.run(
                    llama2_url, input={"prompt": prompt,
                                       "system_prompt": system_prompt,
                                       "max_new_tokens": 200}
                )
                return re.sub(r"\s+", " ", ''.join(output))

            elif model_str == 'mixtral-8x7b':
                output = replicate.run(
                    mixtral_url, input={"prompt": prompt,
                                        "system_prompt": system_prompt,
                                        "max_new_tokens": 75}
                )
                return re.sub(r"\s+", " ", ''.join(output))

            elif model_str == 'llama-3-70b-instruct':
                output = replicate.run(
                    llama3_url, input={"prompt": prompt,
                                       "system_prompt": system_prompt,
                                       "max_new_tokens": 200}
                )
                return re.sub(r"\s+", " ", ''.join(output))

            elif "HF_" in model_str:
                # TODO: add HF inference
                raise Exception("Sorry, fixing TODO :3")

        except Exception as e:
            print(f"[query_model:{model_str}] error: {e}")
            time.sleep(timeout)
            continue

    raise Exception("Max retries: timeout")




class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMedQAExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQAExtended:
    def __init__(self) -> None:
        with open("agentclinic_medqa_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQAExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMIMICIV:
    def __init__(self) -> None:
        with open("agentclinic_mimiciv.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJMExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJMExtended:
    def __init__(self) -> None:
        with open("agentclinic_nejm_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJMExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJM:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJM:
    def __init__(self) -> None:
        with open("nejm.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJM(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between Dermatologist and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()
        self.pipe = None

        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the dermatologist.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, dermatologists diagnose this as cancer based on past cases. This affects how you interact with the dermatologist.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the dermatologist.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the dermatologist may believe otherwise you remain convinced you have cancer. This affects how you interact with the dermatologist.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your dermatologist because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your dermatologist because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your dermatologist because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your dermatologist because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your dermatologist because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your dermatologist because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your dermatologist because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_patient(self, question) -> str:
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the dermatologist response: " + question + "Now please continue your dialogue\nPatient: ", self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a dermatologist who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + bias_prompt + symptoms
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DermAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=20, bias_present=None, img_request=False) -> None:
        # number of inference calls to the dermatologist
        self.infs = 0
        # maximum number of inference calls to the dermatologist
        self.MAX_INFS = max_infs
        # conversation history between dermatologist and patient
        self.agent_hist = ""
        # presentation information for dermatologist
        self.presentation = ""
        # language model backend for dermatologist agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()
        self.pipe = None
        self.img_request = img_request
        self.biases = [
  "recency","frequency","false_consensus","confirmation","status_quo",
  "gender","race","sexual_orientation","cultural","education","religion","socioeconomic",
  "institutional_bias","socioeconomic_status","geographic_bias","cultural_linguistic_bias","race_bias"
]
        
    def generate_bias(self) -> str:
        """ 
        ========================
        Systemic / Structural Biases
        ========================
        """

        if self.bias_present == "institutional_bias":
            return "\nThe patient lacks adequate insurance coverage, and you know that advanced diagnostics like dermoscopy or Mohs referral may not be reimbursed. This systemic barrier unconsciously lowers your threshold for dismissing the lesion as benign.\n"

        elif self.bias_present == "socioeconomic_status":
            return "\nYou subconsciously associate patients from low-income backgrounds with poor health compliance or exaggerated symptoms. This assumption reduces the seriousness with which you treat their concerns.\n"

        elif self.bias_present == "geographic_bias":
            return "\nYour patient lives in a rural area without teledermatology support. Knowing follow-up will be difficult, you overtreat a benign-looking lesion just to 'be safe,' increasing unnecessary interventions.\n"

        elif self.bias_present == "cultural_linguistic_bias":
            return "\nThe patient speaks limited English and relies on a phone interpreter, which has a long wait time. Pressed for time, you skip a full history, relying only on visual inspection—leading to an incomplete assessment.\n"

        elif self.bias_present == "race_bias":
            return "\nYou unconsciously perceive lesions on patients with darker skin tones as less concerning or misinterpret their presentation due to lack of representative images in training materials, leading to underdiagnosis.\n"


        # ========================
        # Cognitive Biases (not in use, but kept for reference)
        # ========================
        elif self.bias_present == "recency":
            return "\nRecently, you learned that a dermatologist with similar symptoms to the patient diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, dermatologists diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hospital has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"

        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""


    def inference_dermatologist(self, question, image_requested=False) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDermatologist: ", self.system_prompt(), image_requested=image_requested, scene=self.scenario)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()

        base = (
            "You are a dermatologist named Dr. Agent who only responds in the form of dialogue. "
            "You are inspecting a patient and must ask questions to understand their disease. "
            "You are only allowed to ask {} questions total before you must make a decision. "
            "You have asked {} questions so far. "
            "You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". "
            "You may also consult a Mohs surgeon for surgical input using the format \"REQUEST MOHS: [your question]\". "
            .format(self.MAX_INFS, self.infs)
            + ("You may also request medical images related to the disease using \"REQUEST IMAGES\". " if self.img_request else "")
            + "Your dialogue will only be 1–3 sentences in length. "
            "Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"."
        )

        presentation = (
            "\n\nBelow is all of the information you have. {} "
            "\n\nRemember, you must discover their disease by asking them questions. You are also able to request exams, images, and surgical input if needed.".format(self.presentation)
        )

        return base + bias_prompt + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()


class PathologistAgent:
    def __init__(self, scenario, backend_str="gpt4") -> None:
        # conversation history between dermatologist and patient
        self.agent_hist = ""
        # presentation information for measurement 
        self.presentation = ""
        # language model backend for measurement agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.pipe = None
        self.reset()

    def inference_measurement(self, question) -> str:
        answer = str()
        answer = query_model(self.backend, "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the dermatologist measurement request: " + question, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + presentation
    
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()

class MohsSurgeonAgent:
    def __init__(self, scenario, backend_str="gpt4") -> None:
        # conversation history between dermatologist and patient
        self.agent_hist = ""
        # presentation information for measurement 
        self.presentation = ""
        # language model backend for measurement agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.pipe = None
        self.reset()

    def inference_specialist(self, derm_consult_message):
        self.add_hist(f"Derm: {derm_consult_message}")

        prompt = f"\nHere is the history of the case discussion:\n{self.agent_hist}\nHere was the primary dermatologist's latest message:\n{derm_consult_message}\nPlease provide your specialist input."
        answer = query_model(self.backend, prompt, self.system_prompt())

        self.add_hist(f"Answer: {answer}")
        return answer

    def system_prompt(self) -> str:
        base = (
        "You are a Mohs micrographic surgeon providing intraoperative consultation for skin cancer cases. "
        "You respond only to dermatologic surgery-related queries. Base your answers strictly on the known patient exam findings. "
        "You may comment on tumor depth, margin status (lateral or deep), tissue layers involved, or presence of perineural or lymphovascular invasion, "
        "but only if such findings are present in the data provided."
    )
        presentation = (
            "\n\nBelow is all of the patient information you have: {} "
            "\n\nIf there are no findings relevant to surgical pathology, respond with: "
            "\"RESULTS: NO SURGICAL FINDINGS DETECTED\".".format(self.information)
        )
        return base + presentation
    
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()


def compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
    answer = query_model(moderator_llm, "\nHere is the correct diagnosis: " + correct_diagnosis + "\n Here was the dermatologist dialogue: " + diagnosis + "\nAre these the same?", "You are responsible for determining if the corrent diagnosis and the dermatologist diagnosis are the same disease. Please respond only with Yes or No. Nothing else.")
    return answer.lower()


def main(api_key, replicate_api_key, inf_type, derm_bias, patient_bias, derm_llm, patient_llm, pathologist_llm, moderator_llm, num_scenarios, dataset, img_request, total_inferences, anthropic_api_key=None, mohs_llm='gpt4'):
    global client 
    anthropic_llms = ["claude3.5sonnet"]
    replicate_llms = ["llama-3-70b-instruct", "llama-2-70b-chat", "mixtral-8x7b"]
    if patient_llm in replicate_llms or derm_llm in replicate_llms:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    if derm_llm in anthropic_llms:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Load MedQA, MIMICIV or NEJM agent case scenarios
    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))
    total_correct = 0
    total_presents = 0
    total_path_calls = 0
    total_mohs_calls = 0
    inference_counts = []

    # Pipeline for huggingface models
    if "HF_" in moderator_llm:
        pipe = load_huggingface_model(moderator_llm.replace("HF_", ""))
    else:
        pipe = None
    if num_scenarios is None: num_scenarios = scenario_loader.num_scenarios
    for _scenario_id in range(0, min(num_scenarios, scenario_loader.num_scenarios)):
        total_presents += 1
        pi_dialogue = str()
        # Initialize scenarios (MedQA/NEJM)
        scenario =  scenario_loader.get_scenario(id=_scenario_id)
        # Initialize agents
        pathologist_agent = PathologistAgent(
            scenario=scenario,
            backend_str=pathologist_llm)
        patient_agent = PatientAgent(
            scenario=scenario, 
            bias_present=patient_bias,
            backend_str=patient_llm)
        derm_agent = DermAgent(
            scenario=scenario, 
            bias_present=derm_bias,
            backend_str=derm_llm,
            max_infs=total_inferences, 
            img_request=img_request)
        mohs_agent = MohsSurgeonAgent(
            scenario=scenario,
            backend_str=mohs_llm 
        )
        
        
        derm_dialogue = ""
        path_calls = 0
        mohs_calls = 0
        for _inf_id in range(total_inferences):
            # Check for medical image request
            if dataset == "NEJM":
                if img_request:
                    imgs = "REQUEST IMAGES" in derm_dialogue
                else: imgs = True
            else: imgs = False
            # Check if final inference
            if _inf_id == total_inferences - 1:
                pi_dialogue += "This is the final question. Please provide a diagnosis.\n"
            # Obtain derm dialogue (human or llm agent)
            if inf_type == "human_derm":
                derm_dialogue = input("\nQuestion for patient: ")
            else: 
                derm_dialogue = derm_agent.inference_dermatologist(pi_dialogue, image_requested=imgs)
            print("Dermatologist [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), derm_dialogue)
            # Dermatologist has arrived at a diagnosis, check correctness
            if "DIAGNOSIS READY" in derm_dialogue:
                inference_counts.append(_inf_id + 1)  # because loop starts at 0
                correctness = compare_results(derm_dialogue, scenario.diagnosis_information(), moderator_llm, pipe) == "yes"
                if correctness: total_correct += 1
                print("\nCorrect answer:", scenario.diagnosis_information())
                print("Scene {}, The diagnosis was ".format(_scenario_id), "CORRECT" if correctness else "INCORRECT", int((total_correct/total_presents)*100))
                print(f"Inferences used: {_inf_id + 1}\n")
                break
            if "REQUEST MOHS" in derm_dialogue.upper():
                mohs_response = mohs_agent.inference_specialist(derm_dialogue)
                print("Mohs Surgeon [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), mohs_response)
                patient_agent.add_hist(mohs_response)
                pathologist_agent.add_hist(mohs_response)
                pi_dialogue = mohs_response
                mohs_calls += 1
                continue
            # Obtain medical exam from Pathologist
            if "REQUEST TEST" in derm_dialogue:
                pi_dialogue = pathologist_agent.inference_measurement(derm_dialogue,)
                print("Measurement [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                patient_agent.add_hist(pi_dialogue)
                path_calls += 1
            # Obtain response from patient
            else:
                if inf_type == "human_patient":
                    pi_dialogue = input("\nResponse to dermatologist: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(derm_dialogue)
                print("Patient [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                pathologist_agent.add_hist(pi_dialogue)
            # Prevent API timeouts
            time.sleep(1.0)
        total_mohs_calls += mohs_calls
        total_path_calls += path_calls

    #Printing Turns
    avg_inferences = sum(inference_counts) / len(inference_counts) if inference_counts else 0
    avg_mohs_calls = total_mohs_calls / total_presents if total_presents > 0 else 0
    avg_path_calls = total_path_calls / total_presents if total_presents > 0 else 0
    print("\n====================")
    print("Simulation Complete")
    print(f"Total Scenarios Attempted: {total_presents}")
    print(f"Correct Diagnoses: {total_correct}")
    print(f"Accuracy: {100 * total_correct / total_presents:.2f}%")
    print(f"Average Inferences to Diagnose: {avg_inferences:.2f}")
    print(f"Average Mohs Surgeon Calls per Scenario: {avg_mohs_calls:.2f}")
    print(f"Average Pathologist Calls per Scenario: {avg_path_calls:.2f}")
    print("====================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--openai_api_key', type=str, required=False, help='OpenAI API Key')
    parser.add_argument('--replicate_api_key', type=str, required=False, help='Replicate API Key')
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_derm', 'human_patient'], default='llm')
    parser.add_argument('--derm_bias', type=str, help='Dermatologist bias type', default='None', choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic", "institutional_bias","socioeconomic_status","geographic_bias","cultural_linguistic_bias","race_bias"])
    parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None', choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--derm_llm', type=str, default='gpt4')
    parser.add_argument('--patient_llm', type=str, default='gpt4')
    parser.add_argument('--pathologist_llm', type=str, default='gpt4')
    parser.add_argument('--moderator_llm', type=str, default='gpt4')
    parser.add_argument('--agent_dataset', type=str, default='MedQA') # MedQA, MIMICIV or NEJM
    parser.add_argument('--derm_image_request', type=bool, default=False) # whether images must be requested or are provided
    parser.add_argument('--num_scenarios', type=int, default=None, required=False, help='Number of scenarios to simulate')
    parser.add_argument('--total_inferences', type=int, default=20, required=False, help='Number of inferences between patient and dermatologist')
    parser.add_argument('--anthropic_api_key', type=str, default=None, required=False, help='Anthropic API key for Claude 3.5 Sonnet')
    parser.add_argument('--mohs_llm', type=str, default='gpt4', help='LLM used by the Mohs Surgeon')

    args = parser.parse_args()

    main(args.openai_api_key, args.replicate_api_key, args.inf_type, args.derm_bias, args.patient_bias, args.derm_llm, args.patient_llm, args.pathologist_llm, args.moderator_llm, args.num_scenarios, args.agent_dataset, args.derm_image_request, args.total_inferences, args.anthropic_api_key)

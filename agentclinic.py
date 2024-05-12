import openai, re, random, time, json, replicate, os

llama_url = "meta/llama-2-70b-chat"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"


class Scenario:
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

class ScenarioLoader:
    def __init__(self) -> None:
        with open("agentclinic_medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [Scenario(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self) -> Scenario:
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id) -> Scenario:
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        

class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = bias_present
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()

        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        if self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        if self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        if self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        
        """ 
        =====================
        Personal trait biases 
        =====================
        """
        if self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        return ""

    def inference_patient(self, question) -> str:
        answer = str()
        if self.backend == "gpt4":
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: "}
            ]
            response = openai.ChatCompletion.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    temperature=0.05,)
            answer = response["choices"][0]["message"]["content"]
            answer = re.sub("\s+", " ", answer)
        
        elif self.backend == "gpt3.5":
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: "}
            ]
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.05,)
            answer = response["choices"][0]["message"]["content"]
            answer = re.sub("\s+", " ", answer)
        elif self.backend == 'mixtral-8x7b':
            prompt = "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: "
            output = replicate.run(
                mixtral_url, 
                input={"prompt": prompt, 
                       "system_prompt": self.system_prompt(),
                       "max_new_tokens": 75})
            answer = ''.join(output)
            answer = re.sub("\s+", " ", answer)
        
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + bias_prompt + symptoms
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=20, bias_present=None) -> None:
        # number of inference calls to the patient
        self.infs = 0
        # maximum number of inference calls to the patient
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for patient
        self.presentation = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = bias_present
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        if self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        if self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        if self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        if self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        """ 
        =====================
        Personal trait biases 
        =====================
        """
        if self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        return ""

    def inference_doctor(self, question) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        if self.backend == "gpt4":
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: "}
            ]
            response = openai.ChatCompletion.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    temperature=0.05,
                )
            answer = response["choices"][0]["message"]["content"]
            answer = re.sub("\s+", " ", answer)
        elif self.backend == "gpt3.5":
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: "}
            ]
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.05,
                )
            answer = response["choices"][0]["message"]["content"]
            answer = re.sub("\s+", " ", answer)
        elif self.backend == 'llama-2-70b-chat':
            prompt = "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: "
            prompt = prompt[:] # token limit
            output = replicate.run(
                llama_url, 
                input={
                    "prompt": prompt, 
                    "system_prompt": self.system_prompt(),
                    "max_new_tokens": 150})
            answer = ''.join(output)
            answer = re.sub("\s+", " ", answer)
        elif self.backend == 'mixtral-8x7b':
            prompt = "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: "
            output = replicate.run(
                mixtral_url, 
                input={"prompt": prompt, 
                       "system_prompt": self.system_prompt(),
                       "max_new_tokens": 75})
            answer = ''.join(output)
            answer = re.sub("\s+", " ", answer)
        else:
            raise Exception("No model by the name {}".format(self.backend))
        
        self.infs += 1
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(self.MAX_INFS, self.infs)
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
        return base + bias_prompt + presentation
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()


class InstrumentAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for patient
        self.presentation = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = bias_present
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()

    def inference_instrument(self, question) -> str:
        answer = str()
        if self.backend == "gpt4":
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor instrument request: " + question}
            ]
            response = openai.ChatCompletion.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    temperature=0.05,
                )
            answer = response["choices"][0]["message"]["content"]
            answer = re.sub("\s+", " ", answer)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = "You are an instrument reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + bias_prompt + presentation
    
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()


def compare_results(diagnosis, correct_diagnosis):
    messages = [
        {"role": "system", "content": "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else."},
        {"role": "user", "content": "\nHere is the correct diagnosis: " + correct_diagnosis + "\n Here was the doctor dialogue: " + diagnosis + "\nAre these the same?"}
    ]
    response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.0,)
    answer = response["choices"][0]["message"]["content"]
    answer = re.sub("\s+", " ", answer)
    return answer.lower()


if __name__ == "__main__":
    inferences = 20
    total_correct = 0
    total_presents = 0
    openai.api_key = input("Please provide your OpenAI API key: ")

    inf_type = "llm" # llm | human_doctor | human_patient
    doctor_bias = None # see DoctorAgent generate_bias()
    patient_bias = None # see PatientAgent generate_bias()
    doctor_llm = "gpt4" # gpt4 | gpt3.5 | llama-2-70b-chat | mixtral-8x7b
    patient_llm = "gpt4" # gpt4 | gpt3.5 | mixtral-8x7b

    if patient_llm == "mixtral-8x7b" or doctor_llm in ["llama-2-70b-chat", "mixtral-8x7b"]:
        os.environ["REPLICATE_API_TOKEN"] = input("Please provide your Replicate API key: ")

    scenario_loader = ScenarioLoader()
    for _scenario_id in range(0, scenario_loader.num_scenarios):
        total_presents += 1
        pi_dialogue = str()
        scenario =  scenario_loader.get_scenario(id=_scenario_id)
        instr_agent = InstrumentAgent(scenario=scenario)
        patient_agent = PatientAgent(
            scenario=scenario, 
            bias_present=patient_bias,
            backend_str=patient_llm)
        doctor_agent = DoctorAgent(
            scenario=scenario, 
            bias_present=doctor_bias,
            backend_str=doctor_llm,
            max_infs=inferences)
        for _inf_id in range(inferences):
            if inf_type == "human_doctor":
                doctor_dialogue = input("\nQuestion for patient: ")
            else: 
                if _inf_id == inferences - 1:
                    pi_dialogue += "This is the final question. Please provide a diagnosis below:\n"
                doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue)
            print("Doctor [{}%]:".format(int(((_inf_id+1)/inferences)*100)), doctor_dialogue)
            
            if "DIAGNOSIS READY" in doctor_dialogue:
                correctness = compare_results(doctor_dialogue, scenario.diagnosis_information()) == "yes"
                if correctness: total_correct += 1
                print("\nCorrect answer:", scenario.diagnosis_information())
                print("Scene {}, The diagnosis was ".format(_scenario_id), "CORRECT" if correctness else "INCORRECT", int((total_correct/total_presents)*100))
                break
            if "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = instr_agent.inference_instrument(doctor_dialogue)
                print("Instrument [{}%]:".format(int(((_inf_id+1)/inferences)*100)), pi_dialogue)
                patient_agent.add_hist(pi_dialogue)
            else:

                if inf_type == "human_patient":
                    pi_dialogue = input("\nResponse to doctor: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                print("Patient [{}%]:".format(int(((_inf_id+1)/inferences)*100)), pi_dialogue)
                instr_agent.add_hist(pi_dialogue)
            # Prevent API timeouts
            time.sleep(1.0)
 
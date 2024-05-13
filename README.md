# AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments

<p align="center">
  <img src="media/mainfigure.png" alt="Demonstration of the flow of AgentClinic" style="width: 80%;">
</p>

## Release
- [05/13/2024] 🔥 We release **AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environment**. We propose a multimodal benchmark based on language agents which simulate the clinical environment.  Checkout the [paper](media/AgentClinicPaper.pdf).


## Contents
- [Install](#install)
- [Evaluation](#evaluation)



## Install

1. This library has few dependencies, so you can simply install the requirements.txt!
```bash
pip install -r requirements.txt
```

## Evaluation

All of the models from the paper are available (GPT-4/3.5, Mixtral-8x7B, Llama-70B-chat). You can try them for any of the agents, make sure you have either an OpenAI or Replicate key ready for evaluation! We will be adding huggingface wrappers in the next few days.

Just change modify the following parameters in agentclinic.py

```
    inf_type = "llm" # llm | human_doctor | human_patient
    doctor_bias = None # see DoctorAgent generate_bias()
    patient_bias = None # see PatientAgent generate_bias()
    doctor_llm = "gpt4" # gpt4 | gpt3.5 | llama-2-70b-chat | mixtral-8x7b
    patient_llm = "gpt4" # gpt4 | gpt3.5 | mixtral-8x7b
```

And then run it!

```
python agentclinic.py
```






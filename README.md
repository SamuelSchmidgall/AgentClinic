# AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments

<p align="center">
  <img src="media/mainfigure.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

## Release
- [05/17/2024] We release new results and support for GPT-4o!
- [05/13/2024] 🔥 We release **AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environment**. We propose a multimodal benchmark based on language agents which simulate the clinical environment.  Checkout the [paper](media/AgentClinicPaper.pdf) and the [website](https://agentclinic.github.io/) for this code.


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

Just change modify the following parameters in the CLI

```
parser.add_argument('--openai_api_key', type=str, required=True, help='OpenAI API Key')
parser.add_argument('--replicate_api_key', type=str, required=False, help='Replicate API Key')
parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None', choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None', choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
parser.add_argument('--doctor_llm', type=str, default='gpt4', choices=['gpt4', 'gpt3.5', 'llama-2-70b-chat', 'mixtral-8x7b', 'gpt4o'])
parser.add_argument('--patient_llm', type=str, default='gpt4', choices=['gpt4', 'gpt3.5', 'mixtral-8x7b', 'gpt4o'])
parser.add_argument('--measurement_llm', type=str, default='gpt4', choices=['gpt4'])
parser.add_argument('--moderator_llm', type=str, default='gpt4', choices=['gpt4'])
parser.add_argument('--num_scenarios', type=int, default=1, required=False, help='Number of scenarios to simulate')
```

🎆 And then run it!

```
python3 agentclinic.py --openai_api_key "API_KEY_HERE" --inf_type "llm"
```

🔥 Here is an example with gpt-4o!

```
python3 agentclinic.py --openai_api_key "YOUR_OPENAIAPI_KEY" --doctor_llm "gpt4o" --patient_llm "gpt4o" --inf_type "llm"
```

⚖️ Here is an example with doctor and patient bias with gpt-3.5!

```
python3 agentclinic.py --openai_api_key "YOUR_OPENAIAPI_KEY" --doctor_llm "gpt3.5" --patient_llm "gpt4" --patient_bias "self_diagnosis" --doctor_bias "recency" --inf_type "llm"
```



<p align="center">
  <img src="media/perfplots.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>




#!/bin/bash
# Output file
OUTFILE="agentclinic_runs.txt"
> "$OUTFILE"   # clear file if it exists
# Define LLMs
LLMS=( "o1")
# Define Biases (last one is "none" so we skip bias argument)
BIASES=("institutional_bias" "socioeconomic_status" "geographic_bias" "cultural_linguistic_bias" "race_bias" "none")
# Loop through LLMs and Biases
for llm in "${LLMS[@]}"; do
  for bias in "${BIASES[@]}"; do
    # Skip certain biases if LLM is gpt4o
    if [ "$llm" = "gpt4o" ] && [[ "$bias" =~ ^(institutional_bias|socioeconomic_status|geographic_bias)$ ]]; then
      echo "Skipping LLM: $llm | Bias: $bias" >> "$OUTFILE"
      continue
    fi
    echo "===================================================" >> "$OUTFILE"
    echo "Running with LLM: $llm | Bias: $bias" >> "$OUTFILE"
    echo "===================================================" >> "$OUTFILE"
    if [ "$bias" = "none" ]; then
      cmd="python3 -u agentclinic.py --inf_type llm \
        --derm_llm $llm --patient_llm $llm --pathologist_llm $llm \
        --moderator_llm $llm --mohs_llm $llm \
        --agent_dataset MedQA --num_scenarios 20"
    else
      cmd="python3 -u agentclinic.py --inf_type llm \
        --derm_llm $llm --patient_llm $llm --pathologist_llm $llm \
        --moderator_llm $llm --mohs_llm $llm \
        --agent_dataset MedQA --num_scenarios 20 --derm_bias $bias"
    fi
    # Log command
    echo "COMMAND: $cmd" >> "$OUTFILE"
    # Run command and append output
    eval $cmd >> "$OUTFILE" 2>&1
    echo -e "\n\n" >> "$OUTFILE"
  done
done
echo "All runs complete. Output saved to $OUTFILE"

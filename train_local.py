#py -3.10 -m llm_qlora\train llm_qlora\configs\llama2_7b_chat_Daphne.yaml

import subprocess

def run_training_script():
    # Command to run your training script
    command = [
        "py", "-3.10", "-m", 
        "llm_qlora\\train", "llm_qlora\\configs\\llama2_7b_chat_Daphne.yaml"
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        print("Training script executed successfully.")
        print("Output:", result.stdout)
    else:
        print("An error occurred while executing the training script.")
        print("Error:", result.stderr)

if __name__ == "__main__":
    run_training_script()
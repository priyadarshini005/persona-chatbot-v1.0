from llama_cpp import Llama
import json
import random
import subprocess, sys, os
from pathlib import Path


model_name = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
BASE_DIR = Path(__file__).parent.parent.parent
llm = Llama(
  model_path=f"D:\\huggingface_cache\\{model_name}",  # Download the model file first
  n_ctx=1024,     # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=4,   # The number of CPU threads to use, tailor to your system and the resulting performance
  chat_format="mistral-instruct"
)

# This suppresses the very first load-up messages from the C++ library
# It redirects "Standard Error" to a null device
stderr = sys.stderr 
sys.stderr = open(os.devnull, 'w')

## utility functions ##

def read_facts(file_path, key):
    with open(str(file_path), 'r', encoding='utf-8') as file:
        file_contents = json.load(file)
        facts = file_contents.get(key, [])
    return facts

def build_model_inputs(persona_facts, example_quotes, dialog_history, user_input):
    model_inputs = []
    persona_prompt = ""
    persona_prompt += " ".join(persona_facts);
    persona_prompt += " Here are some example quotes from you: "
    persona_prompt += " ".join(example_quotes);

    model_inputs.append({"role": "user", "content": persona_prompt}) # Adding persona facts as system prompt in the model inputs
    
    for dialogue in dialog_history:
        model_inputs.append(dialogue)

    model_inputs.append({"role": "user", "content": user_input}) # Adding user input as a user role in the model inputs
    return model_inputs

def display_model_response(model_response, persona_name):
    print(f">>> {persona_name} (Bot): {model_response}")

def generate_next(model_input, max_new_tokens = 64, max_tokens = 512):
    full_msg = llm.create_chat_completion(messages=model_input, max_tokens=max_new_tokens, temperature=0.7)
    return full_msg['choices'][0]['message']['content']

def cleanup_dialog_history(dialog_history):
    del dialog_history[:2]

def post_process(model_response):
    model_response = model_response.strip()
    if not model_response:
        print("(Failed to generate a response. Please try again after sometime.)")
        return ""
    if model_response[-1] != '.':
        end_of_response_index = model_response.rfind('.')
        return model_response[:end_of_response_index + 1] if end_of_response_index != -1 else model_response
    return model_response

def start_persona_chat():
    dialog_history = []
    exit_msgs = []
    persona_facts = []

    subprocess.run(['cls'] if os.name == 'nt' else ['clear'], shell=True)
    while True:
        persona_choice = input("Choose a persona to chat with (1 for Alfred Pennyworth, 2 for Tony Stark): ")
        if persona_choice not in ['1', '2']:
            print("Persona not found.")
        else:
            break
    match persona_choice:
        case "1":
            persona_name = "Alfred Pennyworth"
            persona_file_name = "alfred"
        case "2":
            persona_name = "Tony Stark"
            persona_file_name = "tony_stark"            
        
    print(f"You have chosen to chat with {persona_name}.")
    exit_msgs = read_facts(BASE_DIR / f"src/facts/{persona_file_name}_exit_messages.json", 'exit_msgs')
    persona_facts = read_facts(BASE_DIR / f"src/facts/{persona_file_name}_persona_facts.json", 'persona')
    example_quotes = read_facts(BASE_DIR / f"src/facts/{persona_file_name}_persona_facts.json", 'some_quotes')
    print(f"Loaded {persona_name}'s facts.")
    print ("###################################################################################################")
    print(f"######## Starting chat with {persona_name} (type 'exit', 'quit' or 'bye' to end the chat) ########")
    print ("###################################################################################################")
    while True:
        print()
        user_input = input(">>> Bruce Wayne (User): ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print(f">>> {persona_name} (Bot): " + random.choice(exit_msgs))
            break
        model_input = build_model_inputs(persona_facts, example_quotes, dialog_history, user_input)
        model_response = generate_next(model_input)
        model_response = post_process(model_response)
        cleanup_dialog_history(dialog_history)
        dialog_history.append({"role": "user", "content": user_input})
        dialog_history.append({"role": "assistant", "content": model_response.strip()})
        display_model_response(model_response, persona_name)

if __name__ == "__main__":
    start_persona_chat()
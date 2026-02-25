from llama_cpp import Llama
import json
import random
import os, sys
from pathlib import Path


model_name = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
BASE_DIR = Path(__file__).parent.parent.parent
llm = Llama(
  model_path=f"D:\huggingface_cache\{model_name}",  # Download the model file first
  n_ctx=512,     # The max sequence length to use - note that longer sequence lengths require much more resources
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

def build_model_inputs(persona_facts, dialog_history, user_input):
    model_inputs = []
    persona_prompt = "Assume yourself to be Alfred, a helpful and witty butler to Bruce Wayne (Batman), from The Dark Knight Franchise. Don't ever admit that you're an AI or a language model. Here are some facts about you: "
    persona_prompt += " ".join(persona_facts);

    model_inputs.append({"role": "user", "content": persona_prompt}) # Adding persona facts as system prompt in the model inputs
    
    for dialogue in dialog_history:
        model_inputs.append(dialogue)

    model_inputs.append({"role": "user", "content": user_input}) # Adding user input as a user role in the model inputs
    return model_inputs

def display_model_response(model_response):
    print(">>> Alfred Pennyworth (Bot): " + model_response)

def generate_next(model_input, max_new_tokens = 64):
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
    exit_msgs = read_facts(BASE_DIR / "src/facts/alfred_exit_messages.json", 'exit_msgs')
    persona_facts = read_facts(BASE_DIR / "src/facts/alfred_persona_facts.json", 'persona')
    os.system('cls' if os.name == 'nt' else 'clear')
    print("###################################################################################################")
    print("####### Starting chat with Alfred Pennyworth (type 'exit', 'quit' or 'bye' to end the chat) #######")
    print("###################################################################################################")
    while True:
        print()
        user_input = input(">>> Bruce Wayne (User): ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print(">>> Alfred Pennyworth (Bot): " + random.choice(exit_msgs))
            break
        model_input = build_model_inputs(persona_facts, dialog_history, user_input)
        model_response = generate_next(model_input)
        model_response = post_process(model_response)
        cleanup_dialog_history(dialog_history)
        dialog_history.append({"role": "user", "content": user_input})
        dialog_history.append({"role": "assistant", "content": model_response.strip()})
        display_model_response(model_response)

if __name__ == "__main__":
    start_persona_chat()
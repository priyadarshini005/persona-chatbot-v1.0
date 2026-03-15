import gradio as gr
from llama_cpp import Llama
import json
import random
import subprocess, sys, os
from pathlib import Path
import time

model_name = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
BASE_DIR = Path(__file__).parent.parent.parent

llm = Llama(
    model_path=f"D:\\huggingface_cache\\{model_name}",
    n_ctx=1024,
    n_threads=4,
    chat_format="mistral-instruct"
)

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')


# -------------------------
# Existing utility functions
# -------------------------

def read_facts(file_path, key):
    with open(str(file_path), 'r', encoding='utf-8') as file:
        file_contents = json.load(file)
        facts = file_contents.get(key, [])
    return facts


def build_model_inputs(persona_facts, example_quotes, dialog_history, user_input):
    model_inputs = []

    persona_prompt = ""
    persona_prompt += " ".join(persona_facts)
    persona_prompt += " Here are some example quotes from you: "
    persona_prompt += " ".join(example_quotes)

    model_inputs.append({"role": "user", "content": persona_prompt})

    for dialogue in dialog_history:
        model_inputs.append(dialogue)

    model_inputs.append({"role": "user", "content": user_input})

    return model_inputs


def generate_next(model_input, max_new_tokens=64):
    full_msg = llm.create_chat_completion(
        messages=model_input,
        max_tokens=max_new_tokens,
        temperature=0.7
    )

    return full_msg['choices'][0]['message']['content']


def cleanup_dialog_history(dialog_history):
    if len(dialog_history) >= 2:
        del dialog_history[:2]


def post_process(model_response):
    model_response = model_response.strip()

    if not model_response:
        return "(Failed to generate a response. Please try again.)"

    if model_response[-1] != '.':
        end = model_response.rfind('.')
        if end != -1:
            return model_response[:end + 1]

    return model_response


# -------------------------
# Persona Loader
# -------------------------

def load_persona(persona_key):

    if persona_key == "alfred":
        persona_name = "Alfred Pennyworth"
    else:
        persona_name = "Tony Stark"

    persona_file = persona_key

    exit_msgs = read_facts(
        BASE_DIR / f"src/facts/{persona_file}_exit_messages.json",
        "exit_msgs"
    )

    persona_facts = read_facts(
        BASE_DIR / f"src/facts/{persona_file}_persona_facts.json",
        "persona"
    )

    example_quotes = read_facts(
        BASE_DIR / f"src/facts/{persona_file}_persona_facts.json",
        "some_quotes"
    )

    return persona_name, exit_msgs, persona_facts, example_quotes


# -------------------------
# Chat logic (Gradio)
# -------------------------

def respond(message, history, persona_key, dialog_history, user_label_value, bot_label_value):

    if persona_key is None:
        yield "", history, dialog_history
        return

    if not message.strip():
        yield "", history, dialog_history
        return
    
    history.append({"role": "user", "content": message})
    # history.append({"role": bot_label_value, "content": "• • •"})
    persona_name, exit_msgs, persona_facts, example_quotes = load_persona(persona_key)

    if message.lower() in ["exit", "quit", "bye"]:
        # history = history + [(f"**{user_label_value}:** {message}", f"**{bot_label_value}:** {random.choice(exit_msgs)}")]
        history.append({"role": "assistant", "content": random.choice(exit_msgs)})
        yield "", history, dialog_history
        return

    # typing indicator
    history.append({"role": "assistant", "content": "• • •"})
    yield "", history, dialog_history

    model_input = build_model_inputs(
        persona_facts,
        example_quotes,
        dialog_history,
        message
    )

    model_response = generate_next(model_input)
    model_response = post_process(model_response)

    cleanup_dialog_history(dialog_history)

    dialog_history.append({"role": "user", "content": message})
    dialog_history.append({"role": "assistant", "content": model_response})

    # history[-1] = (f"**{user_label_value}:** {message}", f"**{bot_label_value}:** {model_response}")
    history[-1] = ({"role": "assistant", "content": model_response})

    yield "", history, dialog_history


# -------------------------
# Persona Selection
# -------------------------

def select_persona(persona_key):
    if persona_key == "alfred":
        persona_name = "Alfred Pennyworth"
        user_label_value = "Bruce Wayne (User)"
    elif persona_key == "tony_stark":
        persona_name = "Tony Stark"
        user_label_value = "User"

    bot_label_value = f"{persona_name} (Bot)"
    title = f"""
## Starting chat with {persona_name}
(type 'exit', 'quit' or 'bye' to end the chat)
"""

    return (
        persona_key,
        [],
        [],
        gr.update(visible=False),
        gr.update(visible=True),
        title,
        user_label_value,
        bot_label_value,
        gr.update(placeholder=f"{user_label_value}: Type your message here...")
    )

# -------------------------
# UI
# -------------------------

with gr.Blocks(title="Persona Chatbot") as demo:

    selected_persona = gr.State()
    dialog_history = gr.State(value=[])
    user_label_value = gr.State()
    bot_label_value = gr.State()

    # Persona screen
    with gr.Column(visible=True) as persona_screen:

        gr.Markdown("## Choose a Persona")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Image(value=str(BASE_DIR / f"src/persona_assets/alfred.jpg"), show_label=False, interactive=False, height=250)
                alfred_btn = gr.Button("Alfred Pennyworth", size="lg")
            with gr.Column(scale=1):
                gr.Image(value=str(BASE_DIR / f"src/persona_assets/tony_stark.jpg"), show_label=False, interactive=False, height=250)
                stark_btn = gr.Button("Tony Stark", size="lg")

    # Chat screen
    with gr.Column(visible=False) as chat_screen:

        chat_title = gr.Markdown()

        chatbot = gr.Chatbot(height=450)

        user_message = gr.Textbox(
            placeholder="Type your message here..."
        )

        clear = gr.Button("Clear Chat")

    # persona selection events

    alfred_btn.click(
        lambda: select_persona("alfred"),
        None,
        [selected_persona, chatbot, dialog_history, persona_screen, chat_screen, chat_title, user_label_value, bot_label_value, user_message]
    )

    stark_btn.click(
        lambda: select_persona("tony_stark"),
        None,
        [selected_persona, chatbot, dialog_history, persona_screen, chat_screen, chat_title, user_label_value, bot_label_value, user_message]
    )

    # chat event

    user_message.submit(
        respond,
        [user_message, chatbot, selected_persona, dialog_history, user_label_value, bot_label_value],
        [user_message, chatbot, dialog_history]
    )

    clear.click(
        lambda: ([], []),
        None,
        [chatbot, dialog_history]
    )


demo.launch()
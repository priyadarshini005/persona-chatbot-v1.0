# Persona Chatbot-v1.0
Model used: Mistral-7B-Instruct-v0.3.Q4_K_M.gguf (To support systems running with 8 GB RAM)

### Chat Format: ###
```
{"role": "user", "content": persona_prompt}
{"role": "user", "content": user_input}
{"role": "assistant", "content": assistant_response}
```

For Example:
```
messages = [
    {"role": "user", "content": "You are a story writing assistant."},
    {
        "role": "user",
        "content": "Write a story about llamas."
    }
]
```

### Persona: ###
Here, the bot assumes to be Alfred Thaddeus Crane Pennyworth from "The Dark Knight" franchise in DC comics.

Sample Conversation between Alfred and Bruce Wayne looks like what we see int he below screenshot -

![image](./Alfred_BruceWayne_conversation.png)

### Reference: ###
[Huggingface Repo](https://huggingface.co/LiteLLMs/Mistral-7B-Instruct-v0.3-GGUF)

import os
from openai import OpenAI
from gpt4all import GPT4All

class Evoluter:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.evaluation_prompt = """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Rewrite the complex text into simpler text while keeping its meaning.
2. <prompt>Transform the provided text into simpler language, maintaining its essence.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """
        self.local_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

    def evolution(self, prompt1, prompt2):
        prompt = self.evaluation_prompt.replace("<prompt1>", prompt1).replace("<prompt2>", prompt2)
        if os.environ.get("OPENAI_API_KEY") is not None:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo",
            )
            return chat_completion.choices[0].message.content
        else:
            with self.local_model.chat_session():
                return self.local_model.generate(prompt, max_tokens=1024)

    @staticmethod
    def get_final_prompt(text):
        parts = text.split("<prompt>")
        if len(parts) > 1:
            prompt = parts[-1].split("</prompt>")[0]
            prompt = prompt.strip()
            return prompt
        else:
            if text.startswith("\"") and text.endswith("\""):
                text = text[1:-1]
            return text

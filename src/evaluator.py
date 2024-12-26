import os
import concurrent.futures
import tqdm
from openai import OpenAI
from gpt4all import GPT4All

class Evaluator:
    def __init__(self) -> None:        
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.local_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    def predict(self, prompt: str, user: str) -> int:
        if os.environ.get("OPENAI_API_KEY") is not None:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": user,
                    }
                ],
                model="gpt-3.5-turbo",
            )
            return int(chat_completion.choices[0].message.content == "positive")
        else:
            with self.local_model.chat_session(system_prompt=prompt):
                return int(self.local_model.generate(user, max_tokens=1024) == "positive")
            
        
    
    def accuracy(self, ground_truth: dict, predict: dict) -> float:
        score = 0
        for k in ground_truth:
            score += int(ground_truth[k] == predict[k])
        return score / len(ground_truth)
        
    def batch_predict(self, prompt: str, ground_truth: dict) -> float:
        predict = {}
        # Define a helper function to be run in parallel
        def predict_for_user(user):
            return user, self.predict(prompt=prompt, user=user)

        # Use ThreadPoolExecutor for multithreading
        if os.environ.get("OPENAI_API_KEY") is not None:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(predict_for_user, user): user for user in ground_truth}
                
                for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(ground_truth)):
                    user, prediction = future.result()
                    predict[user] = prediction
        else:
            for user in tqdm.tqdm(ground_truth):
                    prediction = predict_for_user(user)
                    predict[user] = prediction

        accuracy = self.accuracy(ground_truth=ground_truth, predict=predict)
        return accuracy
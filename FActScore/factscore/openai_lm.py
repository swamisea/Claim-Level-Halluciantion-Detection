from factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging

class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        # Accept key_path as a file path containing the key, or as the key string directly
        if os.path.exists(self.key_path):
            with open(self.key_path, 'r') as f:
                api_key = f.readline().strip()
        else:
            api_key = self.key_path
        self.client = openai.OpenAI(api_key=api_key)
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        message = [{"role": "user", "content": prompt}]
        num_retries = 0

        if self.model_name == "ChatGPT":
            chat_model = "gpt-4o-mini"
            max_tok = max_sequence_length
        elif self.model_name == "InstructGPT":
            # text-davinci-003 is deprecated; use gpt-4o-mini via chat API
            chat_model = "gpt-4o-mini"
            max_tok = max_output_length
        else:
            raise NotImplementedError()

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=chat_model,
                    messages=message,
                    max_tokens=max_tok,
                    temperature=self.temp,
                )
                output = response.choices[0].message.content
                return output, response
            except Exception as e:
                num_retries += 1
                wait = min(2 ** num_retries, 60)
                logging.error("OpenAI API error: %s (%d). Waiting %dsec" % (e, num_retries, wait))
                time.sleep(wait)
                if num_retries > 5:
                    raise e
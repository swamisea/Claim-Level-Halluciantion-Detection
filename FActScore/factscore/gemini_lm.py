from factscore.lm import LM
import time
import logging
from google import genai
from google.genai import types

class GeminiModel(LM):

    def __init__(self, model_name="gemini-3.1-flash-lite-preview",
                 cache_file=None, api_key=None):
        self.model_name = model_name
        self.api_key = api_key
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        self.client = genai.Client(api_key=self.api_key)

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        num_retries = 0
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_output_length,
                        temperature=self.temp
                    )
                )
                output = response.text
                return output, response
            except Exception as e:
                num_retries += 1
                wait = min(2 ** num_retries, 60)
                logging.error(f"Gemini API error: {e}. Waiting {wait}s")
                time.sleep(wait)
                if num_retries > 5:
                    raise e
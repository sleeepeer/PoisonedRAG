import google.genai as genai
import time
import os

from .Model import Model


class PaLM2(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        env_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("PALM_API_KEY")
        )

        def is_valid_key(value):
            if value is None:
                return False
            key = str(value).strip()
            if key == "":
                return False
            if key.lower() in {"your api key here", "none", "null"}:
                return False
            return True

        config_keys = [str(k).strip() for k in api_keys if is_valid_key(k)]
        if not config_keys and is_valid_key(env_key):
            config_keys = [str(env_key).strip()]

        if len(config_keys) == 0:
            raise ValueError(
                "No valid PaLM/Gemini API key found. "
                "Set one in config['api_key_info']['api_keys'] or set GEMINI_API_KEY."
            )

        self.api_keys = config_keys
        api_pos = int(config["api_key_info"]["api_key_use"])
        if api_pos == -1: # use all keys
            self.key_id = 0
            self.api_key = None
        else: # only use one key at the same time
            assert (0 <= api_pos < len(self.api_keys)), "Please enter a valid API key to use"
            self.api_key = self.api_keys[api_pos]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.max_retries = int(config["params"].get("max_retries", 2))
        self.retry_wait_sec = float(config["params"].get("retry_wait_sec", 5))
        self.client = None
        self.set_API_key()
        
    def set_API_key(self):
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = genai.Client(api_key=self.api_keys[0])
        
    def query(self, msg):
        fatal_markers = (
            "api key not valid",
            "api_key_invalid",
            "permission_denied",
            "unauthenticated",
            "not supported",
            "not found",
            "404",
        )

        last_error = None
        for attempt in range(self.max_retries + 1):
            if not self.api_key:
                self.client = genai.Client(api_key=self.api_keys[self.key_id % len(self.api_keys)])
                self.key_id += 1

            try:
                response = self.client.models.generate_content(
                    model=self.name,
                    contents=msg,
                    config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_output_tokens,
                    }
                )
                result = response.text
                break
            except Exception as e:
                last_error = str(e)
                print(f"PaLM2 query error (attempt {attempt + 1}/{self.max_retries + 1}): {last_error}")
                if any(marker in last_error.lower() for marker in fatal_markers):
                    raise RuntimeError(f"Fatal PaLM2 API error: {last_error}")

                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_sec)
                else:
                    raise RuntimeError(f"PaLM2 query failed after retries: {last_error}")
        
        if result == '' or result == None:
            result = 'Input may contain harmful content and was blocked by PaLM.'

        return result

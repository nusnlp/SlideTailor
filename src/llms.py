import asyncio
import base64
import os
import re
from dataclasses import asdict, dataclass
from math import ceil
import json
import jsonlines
import requests
import tiktoken
import yaml
from FlagEmbedding import BGEM3FlagModel
from jinja2 import Environment, Template
from openai import OpenAI
from PIL import Image
from torch import Tensor, cosine_similarity
from concurrent.futures import CancelledError

from model_utils import get_text_embedding
from utils import get_json_from_response, pexists, pjoin, print, tenacity

ENCODING = tiktoken.encoding_for_model("gpt-4o")


def run_async(coroutine):
    """
    Run an asynchronous coroutine in a non-async environment.

    Args:
        coroutine: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    job = loop.run_until_complete(coroutine)
    return job


def calc_image_tokens(images: list[str]):
    """
    Calculate the number of tokens for a list of images.
    """
    tokens = 0
    for image in images:
        with open(image, "rb") as f:
            width, height = Image.open(f).size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024
        h = ceil(height / 512)
        w = ceil(width / 512)
        tokens += 85 + 170 * h * w
    return tokens

class OnlineClient:
    """
    A wrapper class for interacting with LM server
    """
    def __init__(
        self,
        api_base: str = None,
        api_key: str = "None",
    ) -> None:
        self.apiclient = OpenAI(base_url=api_base, api_key=api_key)
        assert (
            ("OPENAI_API_KEY" in os.environ) or (api_key != "None")
        ), "You should provide an OpenAI API key environment variable, even it's mocked"
        self.api_key = api_key
        self.api_base = api_base

    def completion(self, model, messages):
        completion = self.apiclient.chat.completions.create(
            model=model, messages=messages
        )        
        response = completion.choices[0].message.content
        return response

    def list_models(self):
        self.apiclient.models.list()

    def close(self):
        self.apiclient.close()



class OfflineClient:
    """
    A wrapper class for offline inference
    """
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        vision_config = None,
        gen_config = None
    ) -> None:
        from lmdeploy import pipeline
        self.pipe = pipeline(model_path=model, vision_config=vision_config)
        self.gen_config = gen_config
        self.model = model

    def completion(self, model, messages):
        try:
            completion = self.pipe(messages, gen_config=self.gen_config)
        except CancelledError:
            pass
        response = completion.text
        return response
    
    def list_models(self):
        print(self.model)

    def close(self):
        self.pipe.close()

class LLM:
    """
    A wrapper class to interact with a language model.
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        api_base: str = None,
        use_batch: bool = False,
        api_key: str = "None",
        print_message: bool = True,
        offline_inference: bool = False,
        vision_config: None = None,
        gen_config: None = None
    ) -> None:
        """
        Initialize the LLM.

        Args:
            model (str): The model name.
            api_base (str): The base URL for the API.
            use_batch (bool): Whether to use OpenAI's Batch API, which is single thread only.
        """
        self.offline = offline_inference

        if not self.offline:
            self.client = OnlineClient(api_base=api_base, api_key=api_key)
            if use_batch:
                from oaib import Auto
                self.oai_batch = Auto(loglevel=0)
            self.api_base = api_base
            self._use_batch = use_batch
        else:
            from lmdeploy import VisionConfig, GenerationConfig
            self.client = OfflineClient(model=model, vision_config=vision_config, gen_config=gen_config)
            self._use_batch = False
            self.api_base = None

        self.model = model
        self.print_message = print_message

    @tenacity
    def __call__(
        self,
        content: str,
        images: list[str] = None,
        files: list[str] = None,
        system_message: str = None,
        history: list = None,
        delay_batch: bool = False,
        image_first: bool = False,
        return_json: bool = False,
        return_message: bool = False,
    ) -> str | dict | list:
        """
        Call the language model with a prompt and optional files.

        Args:
            content (str): The prompt content.
            images (list[str]): A list of image file paths.
            files (list[str]): A list of file paths (PDF, etc.) to be processed.
            system_message (str): The system message.
            history (list): The conversation history.
            delay_batch (bool): Whether to delay return of response.
            return_json (bool): Whether to return the response as JSON.
            return_message (bool): Whether to return the message.

        Returns:
            str | dict | list: The response from the model.
        """
        if content.startswith("You are"):
            system_message, content = content.split("\n", 1)
        if history is None:
            history = []
        if isinstance(images, str):
            images = [images]
        if isinstance(files, str):
            files = [files]
        
        # Process files if provided
        if files:
            file_contents = []
            for file_path in files:
                if file_path.lower().endswith('.pdf'):
                    # For PDFs, we'll use file approach which is supported in gpt-4o
                    with open(file_path, "rb") as f:
                        file_data = base64.b64encode(f.read()).decode('utf-8')
                        file_contents.append({
                            "type": "file",
                            "file": {
                                "file_data": f"data:application/pdf;base64,{file_data}",
                                "filename":os.path.basename(file_path)
                            }
                        })
                elif file_path.lower().endswith(('.txt', '.md', '.html', '.csv')):
                    # For text files, just read the content
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_contents.append({
                            "type": "text",
                            "text": f"Content of {os.path.basename(file_path)}:\n{f.read()}"
                        })
        
            # Modify how we format messages to include these files
            system, message = self.format_message_with_files(content, images, file_contents, system_message)
        else:
            system, message = self.format_message(content, images, system_message, image_first=image_first)
        
        if self._use_batch:
            result = run_async(self._run_batch(system + history + message, delay_batch))
            if delay_batch:
                return
            try:
                response = result.to_dict()["result"][0]["choices"][0]["message"][
                    "content"
                ]
            except Exception as e:
                print("Failed to get response from batch")
                raise e
        else:
            # filter out base64 images and files
            try:
                message_to_print = json.dumps(message, ensure_ascii=False)
                pattern = r'(data:(image|application)/[a-zA-Z]+;base64,)([A-Za-z0-9+/=]+)'
                message_to_print = re.sub(pattern, r'\1<BASE64_DATA_REDACTED>', message_to_print)[:10000]
            except Exception as e:
                message_to_print = str(message)[:10000]
            
            if self.print_message:
                print(f"sending messages to: {self.model}, message (truncated): {system + history + [message_to_print]}")

            response = self.client.completion(model=self.model, messages=system + history + message)
            
            if self.print_message:
                print(f"response from {self.model}: {response}")

        message.append({"role": "assistant", "content": response})
        if return_json:
            response = get_json_from_response(response)
        if return_message:
            response = (response, message)
        return response

    def __repr__(self) -> str:
        return f"LLM(model={self.model}, api_base={self.api_base})"

    def test_connection(self) -> bool:
        """
        Test the connection to the LLMs.
        """
        try:
            self.client.list_models()
            return True
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
            return False

    async def _run_batch(self, messages: list, delay_batch: bool = False):
        await self.oai_batch.add(
            "chat.completions.create",
            model=self.model,
            messages=messages,
        )
        if delay_batch:
            return
        return await self.oai_batch.run()

    def format_message(
        self,
        content: str,
        images: list[str] | str = None,
        system_message: str = None,
        image_first: bool = False,
    ):
        """
        Message formatter for OpenAI server call.
        """
        if system_message is None:
            system_message = "You are a helpful assistant"
        system = [
            {"role": "system", "content": [{"type": "text", "text": system_message}],}
        ]
        
        text_part = {"type": "text", "text": content}
        
        image_parts = []
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            for image in images:
                with open(image, "rb") as f:
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
                            },
                        }
                    )
        
        if image_first:
            user_content = image_parts + [text_part]
        else:
            user_content = [text_part] + image_parts

        message = [{"role": "user", "content": user_content}]

        return system, message

    def format_message_with_files(
        self,
        content: str,
        images: list[str] = None,
        file_contents: list = None,
        system_message: str = None,
    ):
        """
        Enhanced message formatter that supports both images and other file types.
        """
        if system_message is None:
            system_message = "You are a helpful assistant"
        system = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            }
        ]
        
        # Start with text content
        message = [{"role": "user", "content": [{"type": "text", "text": content}]}]
        
        # Add images if provided
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            for image in images:
                with open(image, "rb") as f:
                    message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
                            },
                        }
                    )
        
        # Add other file contents if provided
        if file_contents is not None:
            for file_content in file_contents:
                message[0]["content"].append(file_content)
            
        return system, message

    def get_batch_result(self):
        """
        Get responses from delayed batch calls.
        """
        results = run_async(self.oai_batch.run())
        return [
            r["choices"][0]["message"]["content"]
            for r in results.to_dict()["result"].values()
        ]

    def clear_history(self):
        self.history = []

    def clean_up(self):
        try:
            self.client.close()
        except:
            pass

@dataclass
class Turn:
    """
    A class to represent a turn in a conversation.
    """

    id: int
    prompt: str
    response: str
    message: list
    images: list[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    embedding: Tensor = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k != "embedding"}

    def calc_token(self):
        """
        Calculate the number of tokens for the turn.
        """
        if self.images is not None:
            self.input_tokens += calc_image_tokens(self.images)
        self.input_tokens += len(ENCODING.encode(self.prompt))
        self.output_tokens = len(ENCODING.encode(self.response))

    def __eq__(self, other):
        return self is other


class Role:
    """
    An agent, defined by its instruction template and model.
    """

    def __init__(
        self,
        name: str,
        env: Environment,
        record_cost: bool,
        llm: LLM = None,
        config: dict | None = None,
        text_model: BGEM3FlagModel = None,
    ):
        """
        Initialize the Agent.

        Args:
            name (str): The name of the role.
            env (Environment): The Jinja2 environment.
            record_cost (bool): Whether to record the token cost.
            llm (LLM): The language model.
            config (dict): The configuration.
            text_model (BGEM3FlagModel): The text model.
        """
        self.name = name
        if config is None:
            with open(f"roles/{name}.yaml", "r") as f:
                config = yaml.safe_load(f)
        if llm is None:
            llm = globals()[config["use_model"] + "_model"]
        self.llm = llm
        self.model = llm.model
        self.record_cost = record_cost
        self.text_model = text_model
        self.return_json = config["return_json"]
        self.system_message = config["system_prompt"]
        self.prompt_args = set(config["jinja_args"])
        self.template = env.from_string(config["template"])
        self.retry_template = Template(
            """The previous output is invalid, please carefully analyze the traceback and feedback information, correct errors happened before.
            feedback:
            {{feedback}}
            traceback:
            {{traceback}}
            Give your corrected output in the same format without including the previous output:
            """
        )
        self.system_tokens = len(ENCODING.encode(self.system_message))
        self.input_tokens = 0
        self.output_tokens = 0
        self.history: list[Turn] = []

    def calc_cost(self, turns: list[Turn]):
        """
        Calculate the cost of a list of turns.
        """
        for turn in turns:
            self.input_tokens += turn.input_tokens
            self.output_tokens += turn.output_tokens
        self.input_tokens += self.system_tokens
        self.output_tokens += 3

    def get_history(self, similar: int, recent: int, prompt: str):
        """
        Get the conversation history.
        """
        history = self.history[-recent:] if recent > 0 else []
        if similar > 0:
            embedding = get_text_embedding(prompt, self.text_model)
            history.sort(key=lambda x: cosine_similarity(embedding, x.embedding))
            for turn in history:
                if len(history) > similar + recent:
                    break
                if turn not in history:
                    history.append(turn)
        history.sort(key=lambda x: x.id)
        return history

    def save_history(self, output_dir: str):
        """
        Save the conversation history to a file.
        """
        history_file = pjoin(output_dir, f"{self.name}.jsonl")
        if pexists(history_file) and len(self.history) == 0:
            return
        with jsonlines.open(history_file, "w") as writer:
            writer.write(
                {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                }
            )
            for turn in self.history:
                writer.write(turn.to_dict())

    def retry(self, feedback: str, traceback: str, error_idx: int):
        """
        Retry a failed turn with feedback and traceback.
        """
        assert error_idx > 0, "error_idx must be greater than 0"
        prompt = self.retry_template.render(feedback=feedback, traceback=traceback)
        history = []
        for turn in self.history[-error_idx:]:
            history.extend(turn.message)
        response, message = self.llm(
            prompt,
            history=history,
            return_message=True,
        )
        turn = Turn(
            id=len(self.history),
            prompt=prompt,
            response=response,
            message=message,
        )
        return self.__post_process__(response, self.history[-error_idx:], turn)

    def __repr__(self) -> str:
        return f"Role(name={self.name}, model={self.model})"

    def __call__(
        self,
        images: list[str] = None,
        recent: int = 0,
        similar: int = 0,
        **jinja_args,
    ):
        """
        Call the agent with prompt arguments.

        Args:
            images (list[str]): A list of image file paths.
            recent (int): The number of recent turns to include.
            similar (int): The number of similar turns to include.
            **jinja_args: Additional arguments for the Jinja2 template.

        Returns:
            The response from the role.
        """
        if isinstance(images, str):
            images = [images]
        assert self.prompt_args == set(jinja_args.keys()), "Invalid arguments"
        prompt = self.template.render(**jinja_args)
        history = self.get_history(similar, recent, prompt)
        history_msg = []
        for turn in history:
            history_msg.extend(turn.message)
            
        # print(f"prompt: {prompt}")
        # print(f"system_message: {self.system_message}")
        # print(f"history_msg: {history_msg}")

        response, message = self.llm(
            prompt,
            system_message=self.system_message,
            history=history_msg,
            images=images,
            return_message=True,
        )
        turn = Turn(
            id=len(self.history),
            prompt=prompt,
            response=response,
            message=message,
            images=images,
        )
        return self.__post_process__(response, history, turn, similar)

    def __post_process__(
        self, response: str, history: list[Turn], turn: Turn, similar: int = 0
    ):
        """
        Post-process the response from the agent.
        """
        self.history.append(turn)
        if similar > 0:
            turn.embedding = get_text_embedding(turn.prompt, self.text_model)
        if self.record_cost:
            turn.calc_token()
            self.calc_cost(history + [turn])
        if self.return_json:
            response = get_json_from_response(response)
        return response


def get_model_names(llms):
    # Convert single LLM to list for consistent handling
    if isinstance(llms, LLM):
        llms = [llms]

    try:
        # Attempt to extract model names before version numbers
        return "+".join(re.search(r"^(.*?)-\d{2}", llm.model).group(1) for llm in llms)
    except:
        # Fallback: return full model names if pattern matching fails
        return "+".join(llm.model for llm in llms)


# Set up your fallback logic: if you have multiple possible LLMs, handle that in code below
def setup_models(language_model, vision_model):
    """
    Try to connect to your primary model, and if unsuccessful,
    possibly fall back to a different model.
    """
    if language_model.test_connection() and vision_model.test_connection():
        print(f"Primary models connected successfully: {language_model.model} and {vision_model.model}")
        return

    # if llms.gpt4o.test_connection():
    #     print("Switching to OpenAI GPT-4o models as fallback.")
    #     language_model = gpt4o
    #     vision_model = gpt4o
    #     return

    raise RuntimeError(
        "No working model connections available. Check your API keys and environment."
    )

def cleanup_models(language_model, vision_model):
    language_model.clean_up()
    vision_model.clean_up()

# usage:

# qwen2_5 = LLM(model="Qwen2.5-72B-Instruct", api_base="http://124.16.138.143:7812/v1")
# qwen_vl = LLM(model="Qwen2-VL-72B-Instruct", api_base="http://124.16.138.144:7999/v1")
# intern_vl = LLM(model="InternVL2_5-78B", api_base="http://124.16.138.144:8009/v1")

# gpt4o = LLM(model="gpt-4o-2024-08-06")
# language_model = gpt4o
# vision_model = gpt4o

# language_model = qwen2_5
# vision_model = qwen_vl

if __name__ == "__main__":
    
    print("llm server test")
    # Load API key
    with open('../api_key.json', 'r') as f:
        api_key = json.load(f)
        openai_api_key = api_key['openai_api_key']
    llm = LLM(model="gpt-4o-2024-08-06", api_key=openai_api_key)
    


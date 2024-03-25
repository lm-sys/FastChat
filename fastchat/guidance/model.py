from guidance.models._model import Chat, Instruct
from guidance.models._remote import Remote
import tiktoken


class LiteLLM(Remote):
    def __init__(
        self,
        model,
        tokenizer=None,
        echo=True,
        caching=True,
        api_base=None,
        api_key=None,
        custom_llm_provider=None,
        temperature=0.0,
        max_streaming_tokens=1000,
        **kwargs,
    ):
        """Build a new LiteLLM model object that represents a model in a given state."""
        try:
            import litellm
        except ImportError:
            raise Exception(
                "Please install the litellm package version >= 1.7 using `pip install litellm -U` in order to use guidance.models.LiteLLM!"
            )

        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is LiteLLM:
            raise Exception(
                "The LightLLM class is not meant to be used directly! Please use LiteLLMChat, LiteLLMInstruct, or LiteLLMCompletion depending on the model you are using."
            )

        self.litellm = litellm
        self.api_base = api_base
        self.api_key = api_key
        self.custom_llm_provider = custom_llm_provider
        self.model_name = model

        # we pretend it tokenizes like gpt2 if tiktoken does not know about it... TODO: make this better
        if tokenizer is None:
            try:
                tokenizer = tiktoken.encoding_for_model(model)
            except:
                tokenizer = tiktoken.get_encoding("gpt2")

        super().__init__(
            model,
            tokenizer=tokenizer,
            echo=echo,
            caching=caching,
            temperature=temperature,
            max_streaming_tokens=max_streaming_tokens,
            **kwargs,
        )


class LiteLLMCompletion(LiteLLM, Instruct):
    def _generator(self, prompt, temperature):
        # update our shared data state
        self._reset_shared_data(prompt, temperature)

        try:
            generator = self.litellm.completion(
                api_base=self.api_base,
                api_key=self.api_key,
                custom_llm_provider=self.custom_llm_provider,
                model=self.model_name,
                messages=[
                    {"content": prompt.decode("utf8"), "role": "system"}
                ],  # note that role=system is just ignored by litellm but used by them to match chat syntax
                max_tokens=self.max_streaming_tokens,
                n=1,
                top_p=1,
                temperature=temperature,
                stream=True,
            )
        except Exception as e:  # TODO: add retry logic
            raise e

        for part in generator:
            chunk = part.choices[0].delta.content or ""
            yield chunk.encode("utf8")

"""
title: Gemini Manifold google_genai
id: gemini_manifold_google_genai
description: Manifold function for Gemini Developer API. Supports native image generation, grounding with Google Search and streaming. Uses google-genai.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.9.0
requirements: google-genai==1.8.0
"""

# This is a helper function that provides a manifold for Google's Gemini Studio API.
# Be sure to check out my GitHub repository for more information! Contributions, questions and suggestions are very welcome.

# Supported features:
#   - Native image generation (image output), use "gemini-2.0-flash-exp-image-generation"
#   - Display citations in the front-end.
#   - Image input
#   - Streaming
#   - Grounding with Google Search
#   - Safety settings

# Features that are supported by API but not yet implemented in the manifold:
#   TODO Audio input support.
#   TODO Video input support.
#   TODO PDF (other documents?) input support, __files__ param that is passed to the pipe() func can be used for this.
#   TODO Display usage statistics (token counts)

from google import genai
from google.genai import types
import asyncio
import copy
import io
import json
import mimetypes
import os
import uuid
import aiohttp
import base64
import re
import fnmatch
import sys
from fastapi import Request
from pydantic import BaseModel, Field, SecretStr
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Optional,
    TYPE_CHECKING,
    cast,
)
from open_webui.models.chats import Chats
from open_webui.models.files import FileForm, Files
from open_webui.models.functions import Functions
from open_webui.storage.provider import Storage
from open_webui.utils.logger import stdout_format
from loguru import logger

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler
    from manifold_types import *  # My personal types in a separate file for more robustness.

# FIXME What about other 2.0 models?
# according to https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini
ALLOWED_GROUNDING_MODELS = [
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.0-pro",
]

# To avoid conflict name in the future, here use suffix not in gemini naming pattern.
SEARCH_MODEL_SUFFIX = "++SEARCH"


# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main logger.
log = logger.bind(auditable=False)


class Pipe:
    class Valves(BaseModel):
        GEMINI_API_KEY: str = Field(default="")
        GEMINI_API_BASE_URL: str = Field(
            default="https://generativelanguage.googleapis.com",
            description="The base URL for calling the Gemini API",
        )
        MODEL_WHITELIST: str = Field(
            default="*",
            description="Comma-separated list of allowed model names. Supports `fnmatch` patterns: *, ?, [seq], [!seq]. Default value is * (all models allowed).",
        )
        MODEL_BLACKLIST: Optional[str] = Field(
            default=None,
            description="Comma-separated list of blacklisted model names. Supports `fnmatch` patterns: *, ?, [seq], [!seq]. Default value is None (no blacklist).",
        )
        CACHE_MODELS: bool = Field(
            default=True,
            description="Whether to request models only on first load and when whitelist changes.",
        )
        USE_GROUNDING_SEARCH: bool = Field(
            default=False,
            description="Whether to use Grounding with Google Search. For more info: https://ai.google.dev/gemini-api/docs/grounding",
        )
        # FIXME Show this only when USE_GROUNDING_SEARCH is True
        GROUNDING_DYNAMIC_RETRIEVAL_THRESHOLD: float = Field(
            default=0.3,
            description="See Google AI docs for more information. Only supported for 1.0 and 1.5 models",
        )
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=False, description="Whether to request relaxed safety filtering"
        )
        # FIXME: remove OFF and add TRACE, CRITICAL
        LOG_LEVEL: Literal["INFO", "WARNING", "ERROR", "DEBUG", "OFF"] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )
        USE_FILES_API: bool = Field(
            title="Use Files API",
            default=True,
            description="Save the image files using Open WebUI's API for files.",
        )

    def __init__(self):

        # This hack makes the valves values available to the `__init__` method.
        # TODO: Get the id from the frontmatter instead of hardcoding it.
        valves = Functions.get_function_valves_by_id("gemini_manifold_google_genai")
        self.valves = self.Valves(**(valves if valves else {}))
        # FIXME: Is logging out the API key a bad idea?
        print(
            f"[gemini_manifold] self.valves initialized:\n{self.valves.model_dump_json(indent=2)}"
        )

        # TODO: Add the log handler here, not in `pipes`.

        # Initialize the genai client.
        self.client = self._get_genai_client()
        # Get Google models from the API.
        self.google_models = self._get_genai_models()

        self.models: list["ModelData"] = []
        self.last_whitelist: str = self.valves.MODEL_WHITELIST
        self.last_blacklist = self.valves.MODEL_BLACKLIST

        print(f"[gemini_manifold] Function has been initialized:\n{self.__dict__}")

    async def pipes(self) -> list["ModelData"]:
        """Register all available Google models."""

        self._add_log_handler()

        if not self.google_models:
            error_msg = "Error during getting the models from Google API, check logs."
            return [self._return_error_model(error_msg)]

        print(
            f"Saved whitelist: {self.last_whitelist}\nWhitelist from valves: {self.valves.MODEL_WHITELIST}"
        )

        # Return existing models if all conditions are met and no error models are present
        if (
            self.models
            and self.valves.CACHE_MODELS
            and self.last_whitelist == self.valves.MODEL_WHITELIST
            and self.last_blacklist == self.valves.MODEL_BLACKLIST
            and not any(model["id"] == "error" for model in self.models)
        ):
            log.info("Models are already initialized. Returning the cached list.")
            return self.models

        self.last_whitelist = self.valves.MODEL_WHITELIST
        self.last_blacklist = self.valves.MODEL_BLACKLIST
        whitelist = (
            self.valves.MODEL_WHITELIST.replace(" ", "").split(",")
            if self.valves.MODEL_WHITELIST
            else []
        )
        blacklist = (
            self.valves.MODEL_BLACKLIST.replace(" ", "").split(",")
            if self.valves.MODEL_BLACKLIST
            else []
        )
        models: list["ModelData"] = [
            {
                "id": self._strip_prefix(model.name),
                "name": model.display_name,
            }
            for model in self.google_models
            if model.name
            and model.display_name
            and any(fnmatch.fnmatch(model.name, f"models/{w}") for w in whitelist)
            and not any(fnmatch.fnmatch(model.name, f"models/{b}") for b in blacklist)
        ]
        self.models = models
        log.debug("Registered models:", data=models)

        return models

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: "UserData",
        __request__: Request,
        __event_emitter__: Callable[["Event"], Awaitable[None]],
        __metadata__: dict[str, Any],
    ) -> Optional[str]:

        # TODO: [refac] Move __user__ to self like that also.
        self.__event_emitter__ = __event_emitter__

        if not self.client:
            error_msg = "genai client is not initialized."
            await self._emit_error(error_msg, exception=False)
            return

        model_name = body.get("model")
        if not model_name:
            error_msg = "body object does not contain model name."
            await self._emit_error(error_msg, exception=False)
            return None
        model_name = self._strip_prefix(model_name)

        # TODO Contruct a type for `__metadata__`.
        if "error" in __metadata__["model"]["id"]:
            error_msg = f'There has been an error during model retrival phase: {str(__metadata__["model"])}'
            await self._emit_error(error_msg, exception=False)
            return

        # Get the message history directly from the backend.
        # This allows us to see data about sources and files data.
        chat_id = __metadata__.get("chat_id")
        if not chat_id:
            error_msg = "Chat ID not found in request body or metadata."
            await self._emit_error(error_msg, exception=False)
            return None
        chat = Chats.get_chat_by_id_and_user_id(id=chat_id, user_id=__user__["id"])

        if not chat:
            error_msg = f"Chat with ID {chat_id} not found."
            await self._emit_error(error_msg, exception=False)
            return None

        chat_content: ChatChatModel = chat.chat  # type: ignore
        print(
            json.dumps(self._truncate_long_strings(copy.deepcopy(chat_content)), indent=2, default=str)  # type: ignore
        )

        messages = chat_content.get("messages")
        chat_params = chat_content.get("params")
        system_prompt = chat_params.get("system")
        log.debug(f"System prompt: {system_prompt}")
        contents = self._genai_contents_from_messages(messages)

        turn_content_dict_list: list[dict] = []
        for content in contents:
            truncated_content = self._truncate_long_strings(content.model_dump())
            turn_content_dict_list.append(truncated_content)
        log.debug(
            "list[google.genai.types.Content] object that will be given to the Gemini API:",
            content_list=str(turn_content_dict_list),
        )

        # TODO: Take defaults from the general front-end config.
        # FIXME: If the conversation has used Search, then augment the sysetm prompt by telling the model to not use [] citations in it's own respose.
        gen_content_conf = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=body.get("top_k"),
            max_output_tokens=body.get("max_tokens"),
            stop_sequences=body.get("stop"),
            safety_settings=self._get_safety_settings(model_name),
        )

        gen_content_conf.response_modalities = ["Text"]
        if "gemini-2.0-flash-exp-image-generation" in model_name:
            gen_content_conf.response_modalities.append("Image")
            # FIXME: append to user message instead.
            if gen_content_conf.system_instruction:
                gen_content_conf.system_instruction = None
                log.warning(
                    "Image Generation model does not support the system prompt message! Removing the system prompt."
                )

        if self.valves.USE_GROUNDING_SEARCH:
            if model_name.endswith(SEARCH_MODEL_SUFFIX):
                log.info("Using grounding with Google Search.")
                gs = None
                # Dynamic retrieval only supported for 1.0 and 1.5 models.
                if "1.0" in model_name or "1.5" in model_name:
                    gs = types.GoogleSearchRetrieval(
                        dynamic_retrieval_config=types.DynamicRetrievalConfig(
                            dynamic_threshold=self.valves.GROUNDING_DYNAMIC_RETRIEVAL_THRESHOLD
                        )
                    )
                    gen_content_conf.tools = [types.Tool(google_search_retrieval=gs)]
                else:
                    gs = types.GoogleSearchRetrieval()
                    gen_content_conf.tools = [
                        types.Tool(google_search=types.GoogleSearch())
                    ]

        gen_content_args = {
            "model": model_name.replace(SEARCH_MODEL_SUFFIX, ""),
            "contents": contents,
            "config": gen_content_conf,
        }

        # TODO: Handle errors related to Google Safety Settings feature.
        if body.get("stream", False):
            # Streaming response.
            try:
                res, raw_text = await self._stream_response(
                    gen_content_args, __request__, __user__
                )
            except Exception as e:
                error_msg = f"Error occured during streaming: {str(e)}"
                await self._emit_error(error_msg)
                return None
            emit_sources = None
            for chunk in res:
                # TODO: Call this only on final finish chunk.
                emit_sources = await self._get_chat_completion_event_w_sources(
                    chunk, raw_text
                )
            if emit_sources:
                log.info("Sending chat completion event with the sources.")
                await __event_emitter__(emit_sources)
            return None

        # Non-streaming response.
        if "gemini-2.0-flash-exp-image-generation" in model_name:
            warn_msg = "Non-streaming responses with native image gen are not currently supported! Stay tuned! Please enable streaming."
            await self._emit_error(warn_msg, warning=True)
            return None
        try:
            # FIXME: Make it async.
            # FIXME: Support native image gen here too.
            response = self.client.models.generate_content(**gen_content_args)
        except Exception as e:
            error_msg = f"Content generation error: {str(e)}"
            await self._emit_error(error_msg)
            return None
        if raw_text := response.text:
            # TODO: [refac] unify this logic with streaming response.
            emit_sources = await self._get_chat_completion_event_w_sources(
                response, raw_text
            )
            if emit_sources:
                log.info("Sending chat completion event with the sources.")
                await __event_emitter__(emit_sources)
                return None
        else:
            warn_msg = "Non-stremaing response did not have any text inside it."
            await self._emit_error(warn_msg, warning=True)
            return None
        return response.text

    """
    ---------- Helper methods inside the Pipe class ----------
    """

    """Event emission and error logging"""

    async def _emit_completion(
        self,
        content: Optional[str] = None,
        done: bool = False,
        error: Optional[str] = None,
        sources: Optional[list["Source"]] = None,
    ):
        """Constructs and emits completion event."""
        emission: "ChatCompletionEvent" = {
            "type": "chat:completion",
            "data": {"done": done},
        }
        if content:
            emission["data"]["content"] = content
        if error:
            emission["data"]["error"] = {"detail": error}
        if sources:
            emission["data"]["sources"] = sources
        await self.__event_emitter__(emission)

    async def _emit_error(
        self, error_msg: str, warning: bool = False, exception: bool = True
    ) -> None:
        """Emits an event to the front-end that causes it to display a nice red error message."""
        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
        await self._emit_completion(error=f"\n{error_msg}", done=True)

    def _return_error_model(
        self, error_msg: str, warning: bool = False, exception: bool = True
    ) -> "ModelData":
        """Returns a placeholder model for communicating error inside the pipes method to the front-end."""
        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
        return {
            "id": "error",
            "name": "[gemini_manifold] " + error_msg,
        }

    """ChatModel.chat.messages -> list[genai.types.Content] conversion"""

    def _genai_contents_from_messages(
        self, messages: list["MessageModel"]
    ) -> list[types.Content]:
        """
        Transforms `ChatModel.chat.messages` list into `genai.types.Content`.
        """

        def process_user_message(message: "UserMessageModel") -> list[types.Part]:
            """
            Processes user message, extracts image parts.
            """
            parts: list[types.Part] = []
            if files := message.get("files"):
                for file in files:
                    if file.get("type") == "image":
                        if image_part := self._genai_part_from_image_url(
                            file.get("url")
                        ):
                            parts.append(image_part)
            return parts

        def process_assistant_message(
            message: "AssistantMessageModel",
        ) -> tuple[list[types.Part], str]:
            """
            Processes assistant message sources. Removes citation markers if they exsist.
            """
            parts: list[types.Part] = []
            filtered_content = content
            if sources := message.get("sources"):
                filtered_content = self._remove_citation_markers(content, sources)
                log.debug(
                    "Removed source markers from the assistant message.",
                    new_text=filtered_content,
                )
            return parts, filtered_content

        contents: list[types.Content] = []
        for message in messages:
            content = message.get("content")
            role = message.get("role")
            parts: list[types.Part] = []
            if role == "user":
                parts = process_user_message(cast("UserMessageModel", message))
            elif role == "assistant":
                message = cast("AssistantMessageModel", message)
                if not message.get("done"):
                    continue
                role = "model"
                parts, content = process_assistant_message(message)
            else:
                log.warning(f"{role} is an unsupported role, skipping this message.")
                continue
            parts += self._genai_parts_from_text(content)
            contents.append(types.Content(role=role, parts=parts))

        return contents

    def _genai_part_from_image_url(self, image_url: str) -> Optional[types.Part]:
        """
        Processes an image URL and returns a genai.types.Part object from it
        Handles GCS, data URIs, and standard URLs.
        """
        try:
            if image_url.startswith("gs://"):
                # FIXME: mime type helper would error out here, it only handles filenames.
                return types.Part.from_uri(
                    file_uri=image_url, mime_type=self._get_mime_type(image_url)
                )
            elif image_url.startswith("data:image"):
                match = re.match(r"data:(image/\w+);base64,(.+)", image_url)
                if match:
                    print(match.group(1))
                    print(f"{match.group(2)[:20]}[...]{match.group(2)[-20:]}")
                    return types.Part.from_bytes(
                        data=base64.b64decode(match.group(2)),
                        mime_type=match.group(1),
                    )
                else:
                    raise ValueError("Invalid data URI for image.")
            else:  # Assume standard URL
                # FIXME: mime type helper would error out here too, it only handles filenames.
                return types.Part.from_uri(
                    file_uri=image_url, mime_type=self._get_mime_type(image_url)
                )
        except Exception:
            # TODO: Send warnin toast to user in front-end.
            log.exception("Error processing image URL.", image_url=image_url)
            return None

    def _genai_parts_from_text(self, text: str) -> list[types.Part]:
        """
        Turns raw text into list of genai.types.Parts objects.
        Extracts and converts markdown images to parts, preserving text order.
        """
        parts: list[types.Part] = []
        last_pos = 0

        for match in re.finditer(
            r"!\[.*?\]\((data:(image/[^;]+);base64,([^)]+)|/api/v1/files/([a-f0-9\-]+)/content)\)",
            text,
        ):
            # Add text before the image
            text_segment = text[last_pos : match.start()]
            if text_segment.strip():
                parts.append(types.Part.from_text(text=text_segment))

            # Determine if it's base64 or a file URL
            if match.group(2):  # Base64 encoded image
                try:
                    mime_type = match.group(2)
                    base64_data = match.group(3)
                    log.debug(
                        "Found base64 image link!",
                        mime_type=mime_type,
                        base64_data=match.group(3)[:50] + "...",
                    )
                    image_part = types.Part.from_bytes(
                        data=base64.b64decode(base64_data),
                        mime_type=mime_type,
                    )
                    parts.append(image_part)
                except Exception:
                    log.exception("Error decoding base64 image:")

            elif match.group(4):  # File URL
                log.debug("Found API image link!", id=match.group(4))
                file_id = match.group(4)
                file_model = Files.get_file_by_id(file_id)

                if file_model is None:
                    log.warning("File with this ID not found.", id=file_id)
                    #  Could add placeholder text here if desired
                    continue  # Skip to the next match

                try:
                    # "continue" above ensures that file_model is not None
                    content_type = file_model.meta.get("content_type")  # type: ignore
                    if content_type is None:
                        log.warning(
                            "Content type not found for this file ID.",
                            id=file_id,
                        )
                        continue

                    with open(file_model.path, "rb") as file:  # type: ignore
                        image_data = file.read()

                    image_part = types.Part.from_bytes(
                        data=image_data, mime_type=content_type
                    )
                    parts.append(image_part)

                except FileNotFoundError:
                    log.exception(
                        "File not found on disk for this ID.",
                        id=file_id,
                        path=file_model.path,
                    )
                except KeyError:
                    log.exception(
                        "Metadata error for this file ID: 'content_type' missing.",
                        id=file_id,
                    )
                except Exception:
                    log.exception("Error processing file with this ID", id=file_id)

            last_pos = match.end()

        # Add remaining text
        remaining_text = text[last_pos:]
        if remaining_text.strip():
            parts.append(types.Part.from_text(text=remaining_text))

        return parts

    """Model response streaming"""

    async def _stream_response(
        self, gen_content_args: dict, __request__: Request, __user__: "UserData"
    ) -> tuple[list[types.GenerateContentResponse], str]:
        """
        Streams the response to the front-end using `__event_emitter__` and finally returns the full aggregated response.
        """
        response_stream: AsyncIterator[types.GenerateContentResponse] = (
            await self.client.aio.models.generate_content_stream(**gen_content_args)  # type: ignore
        )
        aggregated_chunks: list[types.GenerateContentResponse] = []
        content = ""

        async for chunk in response_stream:
            aggregated_chunks.append(chunk)
            candidate = self._get_first_candidate(chunk.candidates)
            if not candidate or not candidate.content or not candidate.content.parts:
                log.warning(
                    "candidate, candidate.content or candidate.content.parts is missing. Skipping to the next chunk."
                )
                continue
            for part in candidate.content.parts:
                if text_part := part.text:
                    content += text_part
                elif inline_data := part.inline_data:
                    if img_url := self._process_image_part(
                        inline_data, gen_content_args, __user__, __request__
                    ):
                        content += img_url
                # TODO: Streaming this way loses the support for Filter.stream() method processing for filter plugins.
                await self._emit_completion(content)

        return aggregated_chunks, content

    def _get_first_candidate(
        self, candidates: Optional[list[types.Candidate]]
    ) -> Optional[types.Candidate]:
        """Selects the first candidate, logging a warning if multiple exist."""
        if not candidates:
            log.warning("Received chunk with no candidates, skipping processing.")
            return None
        if len(candidates) > 1:
            log.warning("Multiple candidates found, defaulting to first candidate.")
        return candidates[0]

    def _process_image_part(
        self, inline_data, gen_content_args: dict, user: "UserData", request: Request
    ) -> Optional[str]:
        """Handles image data conversion to markdown."""
        mime_type = inline_data.mime_type
        image_data = inline_data.data

        if self.valves.USE_FILES_API:
            image_url = self._upload_image(
                image_data,
                mime_type,
                gen_content_args.get("model", ""),
                "Not implemented yet. TAKE IT FROM gen_content_args contents",
                user["id"],
                request,
            )
            return f"![Generated Image]({image_url})" if image_url else None
        else:
            encoded = base64.b64encode(image_data).decode()
            return f"![Generated Image](data:{mime_type};base64,{encoded})"

    def _upload_image(
        self,
        image_data: bytes,
        mime_type: str,
        model: str,
        prompt: str,
        user_id: str,
        __request__: Request,
    ) -> str | None:
        """
        Helper method that uploads the generated image to a storage provider configured inside Open WebUI settings.
        Returns the url to uploaded image.
        """
        image_format = mimetypes.guess_extension(mime_type)
        id = str(uuid.uuid4())
        # TODO: Better filename? Prompt as the filename?
        name = os.path.basename(f"generated-image{image_format}")
        imagename = f"{id}_{name}"
        image = io.BytesIO(image_data)
        image_metadata = {
            "model": model,
            "prompt": prompt,
        }

        # Upload the image to user configured storage provider.
        log.info("Uploading the image to the configured storage provider.")
        try:
            contents, image_path = Storage.upload_file(image, imagename)
        except Exception:
            error_msg = f"Error occurred during upload to the storage provider."
            log.exception(error_msg)
            return None
        # Add the image file to files database.
        log.info("Adding the image file to Open WebUI files database.")
        file_item = Files.insert_new_file(
            user_id,
            FileForm(
                id=id,
                filename=name,
                path=image_path,
                meta={
                    "name": name,
                    "content_type": mime_type,
                    "size": len(contents),
                    "data": image_metadata,
                },
            ),
        )
        if not file_item:
            log.warning("Files.insert_new_file did not return anything.")
            return None
        # Get the image url.
        image_url: str = __request__.app.url_path_for(
            "get_file_content_by_id", id=file_item.id
        )
        return image_url

    """Client initialization and model retrival from Google API"""

    def _get_genai_client(self) -> Optional[genai.Client]:
        client = None
        if self.valves.GEMINI_API_KEY:
            http_options = types.HttpOptions(base_url=self.valves.GEMINI_API_BASE_URL)
            try:
                client = genai.Client(
                    api_key=self.valves.GEMINI_API_KEY,
                    http_options=http_options,
                )
                log.info("genai client successfully initialized!")
            except Exception:
                log.exception("genai client initialization failed.")
        else:
            log.error("GEMINI_API_KEY is not set.")
        return client

    def _get_genai_models(self):
        """
        Gets valid Google models from the API.
        Returns a google.genai.pagers.Pager object.
        """
        google_models = []
        if self.client:
            try:
                google_models = self.client.models.list(config={"query_base": True})
            except Exception:
                log.exception("Retriving models from Google API failed.")
        log.info(f"Retrieved {len(google_models)} models from Gemini Developer API.")
        # Filter Google models list down.
        return [
            model
            for model in google_models
            if model.supported_actions and "generateContent" in model.supported_actions
        ]

    def _get_safety_settings(self, model_name: str) -> list[types.SafetySetting]:
        """Get safety settings based on model name and permissive setting."""

        if not self.valves.USE_PERMISSIVE_SAFETY:
            return []

        # Settings supported by most models
        category_threshold_map = {
            types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.OFF,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.OFF,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: types.HarmBlockThreshold.OFF,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: types.HarmBlockThreshold.OFF,
            types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: types.HarmBlockThreshold.BLOCK_NONE,
        }

        # Older models use BLOCK_NONE
        if model_name in [
            "gemini-1.5-pro-001",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash-8b-exp-0827",
            "gemini-1.5-flash-8b-exp-0924",
            "gemini-pro",
            "gemini-1.0-pro",
            "gemini-1.0-pro-001",
        ]:
            for category in category_threshold_map:
                category_threshold_map[category] = types.HarmBlockThreshold.BLOCK_NONE

        # Gemini 2.0 Flash supports CIVIC_INTEGRITY OFF
        if model_name in [
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-exp",
        ]:
            category_threshold_map[types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY] = (
                types.HarmBlockThreshold.OFF
            )

        log.debug(
            f"Safety settings: {str({k.value: v.value for k, v in category_threshold_map.items()})}"
        )

        safety_settings = [
            types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in category_threshold_map.items()
        ]
        return safety_settings

    def _strip_prefix(self, model_name: str) -> str:
        """
        Strip any prefix from the model name up to and including the first '.' or '/'.
        This makes the method generic and adaptable to varying prefixes.
        """
        # TODO: [refac] pointless helper, remove.
        try:
            # Use non-greedy regex to remove everything up to and including the first '.' or '/'
            stripped = re.sub(r"^.*?[./]", "", model_name)
            return stripped
        except Exception:
            error_msg = "Error stripping prefix, using the original model name."
            log.exception(error_msg)
            return model_name

    """Citations"""

    async def _get_chat_completion_event_w_sources(
        self, data: types.GenerateContentResponse, raw_str: str
    ) -> Optional["ChatCompletionEvent"]:
        """Adds citation markers and source references to raw_str based on grounding metadata"""

        async def resolve_url(session: aiohttp.ClientSession, url: str) -> str:
            """Asynchronously resolves a URL by following redirects to get final destination URL
            Returns original URL if resolution fails.
            """
            try:
                async with session.get(url, allow_redirects=True) as response:
                    return str(response.url)
            except Exception:
                log.exception(
                    f"Error resolving URL, returning the original url.", url=url
                )
                return url

        # Early exit if no candidate content exists
        if not data.candidates:
            return None

        candidate = data.candidates[0]
        grounding_metadata = candidate.grounding_metadata if candidate else None
        # Require valid metadata structure to proceed
        if (
            not grounding_metadata
            or not grounding_metadata.grounding_supports
            or not grounding_metadata.grounding_chunks
        ):
            return None

        # TODO: Add the used search queries as status message.
        supports = grounding_metadata.grounding_supports
        grounding_chunks = grounding_metadata.grounding_chunks

        # Collect all available URIs from grounding chunks and format them into a list of dicts
        source_metadatas: list[SourceMetadata] = [
            {"source": g_c.web.uri, "supports": []}
            for g_c in grounding_chunks
            if g_c.web and g_c.web.uri
        ]
        if not source_metadatas:
            log.warning("No URLs were found in grounding_metadata.")
            return None

        log.info("Resolving all the source URLs asyncronously.")
        async with aiohttp.ClientSession() as session:
            # Create list of async URL resolution tasks
            resolution_tasks = [
                resolve_url(session, metadata["source"])
                for metadata in source_metadatas
                if metadata["source"]
            ]
            # Wait for all resolutions to complete
            resolved_uris = await asyncio.gather(*resolution_tasks)
        log.info("URL resolution completed.")
        # Replace original urls with resolved ones and store old urls in a separate key.
        for index, metadata in enumerate(source_metadatas):
            metadata["original_url"] = metadata["source"]
            metadata["source"] = resolved_uris[index]

        encoded_raw_str = raw_str.encode()
        # Starting the injection from the end ensures that end_pos variables
        # do not get shifted by next insertions.
        for support in reversed(supports):
            if not (
                support.grounding_chunk_indices
                and support.confidence_scores
                and support.segment
                and support.segment.end_index
                and support.segment.start_index
                and support.segment.text
            ):
                continue
            end_pos = support.segment.end_index
            indices = support.grounding_chunk_indices
            # Create citation markers from associated chunk indices
            citation_markers = "".join(f"[{index}]" for index in indices)
            encoded_citation_markers = citation_markers.encode()
            encoded_raw_str = (
                encoded_raw_str[:end_pos]
                + encoded_citation_markers
                + encoded_raw_str[end_pos:]
            )
            for indice in indices:
                source_metadatas[indice]["supports"].append(support.model_dump())

        # Structure sources with resolved URLs
        sources_list: list[Source] = [
            {
                "source": {"name": "web_search"},
                "document": [""]
                * len(source_metadatas),  # We do not have website content
                "metadata": [metadata for metadata in source_metadatas],
            }
        ]
        # Prepare final event structure with modified content and sources
        event_data: "ChatCompletionEventData" = {
            "content": encoded_raw_str.decode(),  # Content with added citations
            "sources": sources_list,  # List of resolved source references
            "done": True,
        }
        event: "Event" = {
            "type": "chat:completion",
            "data": event_data,
        }
        return event

    def _remove_citation_markers(self, text: str, sources: list["Source"]) -> str:
        processed: set[str] = set()
        for source in sources:
            supports = [
                metadata["supports"]
                for metadata in source.get("metadata", [])
                if "supports" in metadata
            ]
            supports = [item for sublist in supports for item in sublist]
            for support in supports:
                support = types.GroundingSupport(**support)
                indices = support.grounding_chunk_indices
                segment = support.segment
                if not (indices and segment):
                    continue
                segment_text = segment.text
                if not segment_text:
                    continue
                # Using a shortened version because user could edit the assistant message in the front-end.
                # If citation segment get's edited, then the markers would not be removed. Shortening reduces the
                # chances of this happening.
                segment_end = segment_text[-32:]
                if segment_end in processed:
                    continue
                processed.add(segment_end)
                citation_markers = "".join(f"[{index}]" for index in indices)
                # Find the position of the citation markers in the text
                pos = text.find(segment_text + citation_markers)
                if pos != -1:
                    # Remove the citation markers
                    text = (
                        text[: pos + len(segment_text)]
                        + text[pos + len(segment_text) + len(citation_markers) :]
                    )
        return text

    """Other helpers"""

    def _truncate_long_strings(
        self, data: dict[str, Any], max_length: int = 50
    ) -> dict[str, Any]:
        """
        Recursively truncates all string and bytes fields within a dictionary that exceed
        the specified maximum length. Bytes are converted to strings before truncation.

        Args:
            data: A dictionary representing the JSON data.
            max_length: The maximum length of strings before truncation.

        Returns:
            The modified dictionary with long strings truncated.
        """
        for key, value in data.items():
            if isinstance(value, str) and len(value) > max_length:
                truncated_length = len(value) - max_length
                data[key] = value[:max_length] + f"[{truncated_length} chars truncated]"
            elif isinstance(value, bytes) and len(value) > max_length:
                truncated_length = len(value) - max_length
                data[key] = (
                    value.hex()[:max_length]
                    + f"[{truncated_length} hex chars truncated]"
                )
            elif isinstance(value, dict):
                self._truncate_long_strings(
                    value, max_length
                )  # Recursive call for nested dictionaries
            elif isinstance(value, list):
                # Iterate through the list and process each element if it's a dictionary
                for item in value:
                    if isinstance(item, dict):
                        self._truncate_long_strings(item, max_length)

        return data

    def _add_log_handler(self):
        """Adds handler to the root loguru instance for this plugin if one does not exist already."""

        def plugin_filter(record: "Record"):
            """Filter function to only allow logs from this plugin (based on module name)."""
            return record["name"] == __name__  # Filter by module name

        # Access the internal state of the logger
        handlers: dict[int, "Handler"] = logger._core.handlers  # type: ignore
        for key, handler in handlers.items():
            existing_filter = handler._filter
            if (
                hasattr(existing_filter, "__name__")
                and existing_filter.__name__ == plugin_filter.__name__
                and hasattr(existing_filter, "__module__")
                and existing_filter.__module__ == plugin_filter.__module__
            ):
                log.debug("Handler for this plugin is already present!")
                return

        logger.add(
            sys.stdout,
            level=self.valves.LOG_LEVEL,
            format=stdout_format,
            filter=plugin_filter,
        )
        log.info(
            f"Added new handler to loguru with level {self.valves.LOG_LEVEL} and filter {__name__}."
        )

    def _get_mime_type(self, file_uri: str) -> str:
        """
        Determines MIME type based on file extension using the mimetypes module.
        """
        mime_type, encoding = mimetypes.guess_type(file_uri)
        if mime_type is None:
            return "application/octet-stream"  # Default MIME type if unknown
        return mime_type

"""
title: Thinking Gemini
author: suurt8ll
repo_url: https://github.com/suurt8ll/thinking_gemini_open_webui
version: 1.0.0
license: MIT
"""

# Most of the code is from https://openwebui.com/f/matthewh/google_genai
# Go to [repo_url] for issues and suggestions.

import os
import re
import asyncio
import time
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, GenerateContentResponse
from typing import List, Union, Iterator, Callable, Awaitable
from markdown import Markdown

DEBUG = False


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(default="")
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)
        EMIT_INTERVAL: int = Field(
            default=5, description="Interval in seconds between status updates."
        )
        EMIT_STATUS_UPDATES: bool = Field(
            default=False, description="Whether to emit status updates."
        )

    def __init__(self):
        try:
            self.id = "google_genai"
            self.type = "manifold"
            self.name = "Google: "
            self.valves = self.Valves(
                **{
                    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
                    "USE_PERMISSIVE_SAFETY": False,
                    "EMIT_INTERVAL": 1,  # Default emit interval
                    "EMIT_STATUS_UPDATES": os.getenv(
                        "EMIT_STATUS_UPDATES", "False"
                    ).lower()
                    in ["true", "1", "yes"],
                }
            )
            if DEBUG:
                print("[INIT] Initialized Pipe with Valves configuration.")
                print(f"  EMIT_STATUS_UPDATES: {self.valves.EMIT_STATUS_UPDATES}")
        except Exception as e:
            if DEBUG:
                print(f"[INIT] Error during initialization: {e}")
        finally:
            if DEBUG:
                print("[INIT] Initialization complete.")

    async def emit_thoughts(
        self, thoughts: str, __event_emitter__: Callable[[dict], Awaitable[None]]
    ) -> None:
        """Emit thoughts in a collapsible element."""
        try:
            if not thoughts.strip():
                if DEBUG:
                    print("[emit_thoughts] No thoughts to emit.")
                return
            enclosure = f"""<details>
<summary>Click to expand thoughts</summary>
{thoughts.strip()}
</details>""".strip()
            if DEBUG:
                print(f"[emit_thoughts] Emitting thoughts: {enclosure}")
            message_event = {
                "type": "message",
                "data": {"content": enclosure},
            }
            await __event_emitter__(message_event)
        except Exception as e:
            if DEBUG:
                print(f"[emit_thoughts] Error emitting thoughts: {e}")
        finally:
            if DEBUG:
                print("[emit_thoughts] Finished emitting thoughts.")

    def get_google_models(self):
        """Retrieve Google models with prefix stripping."""
        try:
            if not self.valves.GOOGLE_API_KEY:
                if DEBUG:
                    print("[get_google_models] GOOGLE_API_KEY is not set.")
                return [
                    {
                        "id": "error",
                        "name": "GOOGLE_API_KEY is not set. Please update the API Key in the valves.",
                    }
                ]
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            models = genai.list_models()
            models_list = list(models)
            print(len(models_list))
            if DEBUG:
                print(
                    f"[get_google_models] Retrieved {len(models_list)} models from Google."
                )
            filtered_models = [
                {
                    "id": self.strip_prefix(model.name),
                    "name": model.display_name,
                }
                for model in models_list
                if "generateContent" in model.supported_generation_methods
                if model.name.startswith("models/")
                and self.strip_prefix(model.name) == "gemini-2.0-flash-thinking-exp"
            ]
            return filtered_models
        except Exception as e:
            if DEBUG:
                print(f"[get_google_models] Error fetching Google models: {e}")
            return [
                {"id": "error", "name": f"Could not fetch models from Google: {str(e)}"}
            ]
        finally:
            if DEBUG:
                print("[get_google_models] Completed fetching Google models.")

    def strip_prefix(self, model_name: str) -> str:
        """
        Strip any prefix from the model name up to and including the first '.' or '/'.
        This makes the method generic and adaptable to varying prefixes.
        """
        try:
            # Use non-greedy regex to remove everything up to and including the first '.' or '/'
            stripped = re.sub(r"^.*?[./]", "", model_name)
            if DEBUG:
                print(
                    f"[strip_prefix] Stripped prefix: '{stripped}' from '{model_name}'"
                )
            return stripped
        except Exception as e:
            if DEBUG:
                print(f"[strip_prefix] Error stripping prefix: {e}")
            return model_name  # Return original if stripping fails
        finally:
            if DEBUG:
                print("[strip_prefix] Completed prefix stripping.")

    def pipes(self) -> List[dict]:
        """Register all available Google models."""
        try:
            models = self.get_google_models()
            if DEBUG:
                print(f"[pipes] Registered models: {models}")
            return models
        except Exception as e:
            if DEBUG:
                print(f"[pipes] Error in pipes method: {e}")
            return []
        finally:
            if DEBUG:
                print("[pipes] Completed pipes method.")

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> None:  # Changed return type to None
        """Main pipe method to process incoming requests."""
        try:
            if not self.valves.GOOGLE_API_KEY:
                if DEBUG:
                    print("[pipe] GOOGLE_API_KEY is not set.")
                await self.emit_error(__event_emitter__, "GOOGLE_API_KEY is not set")
                return
            try:
                genai.configure(api_key=self.valves.GOOGLE_API_KEY)
                if DEBUG:
                    print("[pipe] Configured Google Generative AI with API key.")
            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error configuring Google Generative AI: {e}")
                await self.emit_error(
                    __event_emitter__, f"Error configuring Google Generative AI: {e}"
                )
                return
            model_id = body.get("model", "")
            if DEBUG:
                print(f"[pipe] Received model ID: '{model_id}'")
            # Updated prefix stripping logic using the generic strip_prefix method
            try:
                model_id = self.strip_prefix(model_id)
                if DEBUG:
                    print(f"[pipe] Stripped model ID: '{model_id}'")
            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error processing model ID: {e}")
                await self.emit_error(
                    __event_emitter__, f"Error processing model ID: {e}"
                )
                return
            if model_id != "gemini-2.0-flash-thinking-exp":
                if DEBUG:
                    print(
                        f"[pipe] Invalid model name: '{model_id}'. Only 'gemini-2.0-flash-thinking-exp' is supported."
                    )
                await self.emit_error(
                    __event_emitter__,
                    f"Error: Invalid model name: {model_id}. Only 'gemini-2.0-flash-thinking-exp' is supported.",
                )
                return
            messages = body.get("messages", [])
            if DEBUG:
                print(f"[pipe] Incoming messages: {messages}")
            # Extract system message if present
            system_message = next(
                (msg["content"] for msg in messages if msg.get("role") == "system"),
                None,
            )
            if DEBUG and system_message:
                print(f"[pipe] Extracted system message: '{system_message}'")
            contents = []
            try:
                for message in messages:
                    if message.get("role") != "system":
                        content = message.get("content", "")
                        if isinstance(content, list):
                            parts = []
                            for item in content:
                                if item.get("type") == "text":
                                    parts.append({"text": item.get("text", "")})
                                elif item.get("type") == "image_url":
                                    image_url = item.get("image_url", {}).get("url", "")
                                    if image_url.startswith("data:image"):
                                        image_data = (
                                            image_url.split(",", 1)[1]
                                            if "," in image_url
                                            else ""
                                        )
                                        parts.append(
                                            {
                                                "inline_data": {
                                                    "mime_type": "image/jpeg",
                                                    "data": image_data,
                                                }
                                            }
                                        )
                                    else:
                                        parts.append({"image_url": image_url})
                            contents.append(
                                {"role": message.get("role"), "parts": parts}
                            )
                        else:
                            role = "user" if message.get("role") == "user" else "model"
                            contents.append(
                                {
                                    "role": role,
                                    "parts": [{"text": content}],
                                }
                            )
                if DEBUG:
                    print(f"[pipe] Processed contents: {contents}")
            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error processing messages: {e}")
                await self.emit_error(
                    __event_emitter__, f"Error processing messages: {e}"
                )
                return
            # Insert system message at the beginning if present
            if system_message:
                try:
                    contents.insert(
                        0,
                        {
                            "role": "user",
                            "parts": [{"text": f"System: {system_message}"}],
                        },
                    )
                    if DEBUG:
                        print("[pipe] Inserted system message into contents.")
                except Exception as e:
                    if DEBUG:
                        print(f"[pipe] Error inserting system message: {e}")
                    await self.emit_error(
                        __event_emitter__, f"Error inserting system message: {e}"
                    )
                    return
            try:
                client = genai.GenerativeModel(model_name=model_id)
                if DEBUG:
                    print(
                        f"[pipe] Initialized GenerativeModel with model ID: '{model_id}'"
                    )
            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error initializing GenerativeModel: {e}")
                await self.emit_error(
                    __event_emitter__, f"Error initializing GenerativeModel: {e}"
                )
                return
            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )
            try:
                if self.valves.USE_PERMISSIVE_SAFETY:
                    safety_settings = {
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                    if DEBUG:
                        print("[pipe] Using permissive safety settings.")
                else:
                    safety_settings = body.get("safety_settings", {})
                    if DEBUG:
                        print("[pipe] Using custom safety settings.")
            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error setting safety settings: {e}")
                await self.emit_error(
                    __event_emitter__, f"Error setting safety settings: {e}"
                )
                return
            if DEBUG:
                print("Google API request details:")
                print("  Model:", model_id)
                print("  Contents:", contents)
                print("  Generation Config:", generation_config)
                print("  Safety Settings:", safety_settings)
            # Initialize timer variables
            thinking_timer_task = None
            start_time = None

            async def thinking_timer():
                """Asynchronous task to emit periodic status updates."""
                elapsed = 0
                try:
                    while True:
                        await asyncio.sleep(self.valves.EMIT_INTERVAL)
                        elapsed += self.valves.EMIT_INTERVAL
                        # Format elapsed time
                        if elapsed < 60:
                            time_str = f"{elapsed}s"
                        else:
                            minutes, seconds = divmod(elapsed, 60)
                            time_str = f"{minutes}m {seconds}s"
                        status_message = f"Thinking... ({time_str} elapsed)"
                        await emit_status(__event_emitter__, status_message, done=False)
                except asyncio.CancelledError:
                    if DEBUG:
                        print("[thinking_timer] Timer task cancelled.")
                except Exception as e:
                    if DEBUG:
                        print(f"[thinking_timer] Error in timer task: {e}")

            async def emit_status(event_emitter, message, done):
                """Emit status updates asynchronously."""
                try:
                    if self.valves.EMIT_STATUS_UPDATES and event_emitter:
                        status_event = {
                            "type": "status",
                            "data": {"description": message, "done": done},
                        }
                        if asyncio.iscoroutinefunction(event_emitter):
                            await event_emitter(status_event)
                        else:
                            # If the emitter is synchronous, run it in the event loop
                            loop = asyncio.get_event_loop()
                            loop.call_soon_threadsafe(event_emitter, status_event)
                        if DEBUG:
                            print(
                                f"[emit_status] Emitted status: '{message}', done={done}"
                            )
                    else:
                        if DEBUG:
                            print(
                                f"[emit_status] EMIT_STATUS_UPDATES is disabled. Skipping status: '{message}'"
                            )
                except Exception as e:
                    if DEBUG:
                        print(f"[emit_status] Error emitting status: {e}")
                finally:
                    if DEBUG:
                        print("[emit_status] Finished emitting status.")

            async def emit_message(event_emitter, content):
                """Emit a message event."""
                try:
                    if event_emitter:
                        message_event = {
                            "type": "message",
                            "data": {"content": content},
                        }
                        if asyncio.iscoroutinefunction(event_emitter):
                            await event_emitter(message_event)
                        else:
                            loop = asyncio.get_event_loop()
                            loop.call_soon_threadsafe(event_emitter, message_event)
                        if DEBUG:
                            print(
                                f"[emit_message] Emitted content: '{content[:50]}...'"
                            )  # Print first 50 chars
                except Exception as e:
                    if DEBUG:
                        print(f"[emit_message] Error emitting message: {e}")

            async def emit_error(event_emitter, error_message):
                """Emit an error message."""
                await emit_message(event_emitter, f"Error: {error_message}")

            try:
                # Emit initial 'Thinking' status
                if self.valves.EMIT_STATUS_UPDATES and __event_emitter__:
                    await emit_status(__event_emitter__, "Thinking...", done=False)
                # Record the start time
                start_time = time.time()
                # Start the thinking timer
                if self.valves.EMIT_STATUS_UPDATES:
                    thinking_timer_task = asyncio.create_task(thinking_timer())

                # Define a helper function to call generate_content
                def generate_content_sync(
                    client, contents, generation_config, safety_settings
                ):
                    return client.generate_content(
                        contents,
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                    )

                # Execute generate_content asynchronously to prevent blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    generate_content_sync,
                    client,
                    contents,
                    generation_config,
                    safety_settings,
                )
                # Process response
                if len(response.candidates[0].content.parts) > 1:
                    thoughts = response.candidates[0].content.parts[0].text
                    answer = response.candidates[0].content.parts[1].text
                    if __event_emitter__:
                        await self.emit_thoughts(thoughts, __event_emitter__)
                        await emit_message(__event_emitter__, answer)  # Emit the answer
                else:
                    result = response.candidates[0].content.parts[0].text
                    await emit_message(__event_emitter__, result)  # Emit the answer
            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error during thinking model processing: {e}")
                await emit_error(__event_emitter__, str(e))
            finally:
                # Calculate total elapsed time
                if start_time:
                    total_elapsed = int(time.time() - start_time)
                    if total_elapsed < 60:
                        total_time_str = f"{total_elapsed}s"
                    else:
                        minutes, seconds = divmod(total_elapsed, 60)
                        total_time_str = f"{minutes}m {seconds}s"
                    # Cancel the timer task
                    if thinking_timer_task:
                        thinking_timer_task.cancel()
                        try:
                            await thinking_timer_task
                        except asyncio.CancelledError:
                            if DEBUG:
                                print("[pipe] Timer task successfully cancelled.")
                        except Exception as e:
                            if DEBUG:
                                print(f"[pipe] Error cancelling timer task: {e}")
                    # Emit final status message
                    if self.valves.EMIT_STATUS_UPDATES:
                        final_status = f"Thinking completed in {total_time_str}."
                        await emit_status(__event_emitter__, final_status, done=True)
        finally:
            pass

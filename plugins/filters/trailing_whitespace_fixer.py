"""
title: Trailing Whitespace Fixer
description: Fixes a glitch where extraneous trailing whitespace is added to user messages, cleaning up the entire conversation history before it reaches the LLM.
id: trailing_whitespace_fixer
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.0.0
"""

import datetime
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    # This ensures type hints are available for development without causing runtime errors.
    from utils.manifold_types import *


class Filter:
    def __init__(self):
        self._log("Function has been initialized!")

    def _log(self, message: str):
        """Helper method for logging with timestamp and caller info."""
        timestamp = datetime.datetime.now().isoformat()
        caller_name = inspect.stack()[1].function
        print(f"[{timestamp}] [{__name__}.{caller_name}] {message}")

    async def inlet(
        self,
        body: "Body",
        __id__: str,
        __metadata__: "Metadata",
        __user__: "UserData",
        __request__: Request,
        __model__: dict,
        __event_emitter__: Callable[["Event"], Awaitable[None]],
        __event_call__: Callable[["Event"], Awaitable[Any]],
    ) -> "Body":
        """
        This method is called before the request is sent to the LLM.
        It cleans trailing whitespace from all user messages in the history
        to fix a recurring formatting glitch.
        """
        self._log("Scanning user messages for trailing whitespace...")

        modified_count = 0
        user_message_count = 0

        if messages := body.get("messages"):
            for message in messages:
                if message.get("role") == "user":
                    user_message_count += 1
                    content = message.get("content")
                    content_parts = []
                    was_modified = False

                    # Unify content processing by coercing strings into a list format.
                    if isinstance(content, str):
                        content_parts = [{"type": "text", "text": content}]
                    elif isinstance(content, list):
                        content_parts = content
                    else:
                        # Skip this message if content is empty or of an unexpected type.
                        continue

                    # Apply the fix to all text parts.
                    for part in content_parts:
                        if part.get("type") == "text" and "text" in part:
                            original_text = part["text"]

                            # The core logic: split by newline, strip trailing space from each line, then rejoin.
                            # This is robust and preserves intentional leading whitespace for code/markdown.
                            lines = original_text.split("\n")
                            stripped_lines = [line.rstrip() for line in lines]
                            fixed_text = "\n".join(stripped_lines)

                            if original_text != fixed_text:
                                part["text"] = fixed_text
                                was_modified = True

                    # If any part of the message was changed, update the message content.
                    if was_modified:
                        message["content"] = content_parts # type: ignore
                        modified_count += 1

        if user_message_count > 0:
            self._log(
                f"Scan complete. Cleaned {modified_count} of {user_message_count} user messages."
            )
        else:
            self._log("Scan complete. No user messages found in payload.")

        return body

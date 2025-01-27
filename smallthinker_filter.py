"""
title: Smallthinker Filter
description: Manifold function for Gemini Developer API. Uses google-genai, supports thinking models.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.4.0
"""

from pydantic import BaseModel, Field
import json
from time import time


class Filter:
    class Valves(BaseModel):
        # TODO Add reasonable default values to the Valve fields.
        # TODO Allow user to define custom tags for thinking models.
        # TODO Allow user to define whitelist of models that will be filtered.
        ENABLE_FORMATTING: bool = Field(default=True)
        SPLIT_KEYWORDS_REMOVE: str = Field(
            default="**Final Answer**",
            description="Comma-separated list of keywords to split and remove from the message content",
        )
        SPLIT_KEYWORDS_KEEP: str = Field(
            default="In conclusion",
            description="Comma-separated list of keywords to split and keep in the message content",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.start_time = 0

    def inlet(self, body: dict, **kwargs) -> dict:
        self.start_time = time()
        return body

    def outlet(self, body: dict, **kwargs) -> dict:
        # TODO Reformat function to be very general, not just for smallthinker:3b.
        # TODO Check if streaming is enabled and if so, dont filter the response if the model has thinking tags.
        # print("`body` object before filtering:")
        # print(json.dumps(body, indent=4))
        if self.valves.ENABLE_FORMATTING:
            for message in body["messages"]:
                if message["role"] == "assistant":
                    content = message["content"]
                    formatted_content = ""
                    thoughts = ""
                    final_response = content

                    # 1. Check for thinking tags
                    start_tag = "<think>"
                    end_tag = "</think>"
                    start_index = content.find(start_tag)
                    end_index = content.rfind(end_tag)

                    if (
                        start_index != -1
                        and end_index != -1
                        and start_index < end_index
                    ):
                        thoughts = content[
                            start_index + len(start_tag) : end_index
                        ].strip()
                        final_response = (
                            content[:start_index].strip()
                            + content[end_index + len(end_tag) :].strip()
                        )

                    else:
                        # 2. Check for split keywords to remove
                        split_keywords_remove = [
                            keyword.strip()
                            for keyword in self.valves.SPLIT_KEYWORDS_REMOVE.split(",")
                        ]
                        split_index_remove = -1
                        keyword_removed_used = None
                        for keyword in split_keywords_remove:
                            index = final_response.rfind(keyword)
                            if index > split_index_remove:
                                split_index_remove = index
                                keyword_removed_used = keyword

                        if split_index_remove != -1:
                            thoughts = final_response[:split_index_remove].strip()
                            final_response = final_response[
                                split_index_remove + len(keyword_removed_used or "") :
                            ].strip()
                        else:
                            # 3. Check for split keywords to keep
                            split_keywords_keep = [
                                keyword.strip()
                                for keyword in self.valves.SPLIT_KEYWORDS_KEEP.split(
                                    ","
                                )
                            ]
                            split_index_keep = -1
                            for keyword in split_keywords_keep:
                                index = final_response.rfind(keyword)
                                if (
                                    index != -1
                                ):  # Find first occurence for keywords to keep.
                                    split_index_keep = index
                                    break  # Stop at the first match for keywords to keep

                            if split_index_keep != -1:
                                thoughts = final_response[:split_index_keep].strip()
                                final_response = final_response[
                                    split_index_keep:
                                ].strip()

                    if thoughts:
                        # TODO Use the exact same formatting as Open WebUI.
                        formatted_content += (
                            "<details>\n<summary>Thought for {:.0f} seconds</summary>\n".format(
                                time() - self.start_time
                            )
                            + "".join([f"> {line}\n" for line in thoughts.splitlines()])
                            + "</details>\n"  # Added newline for better separation
                        )
                    if final_response:
                        formatted_content += final_response.strip()
                    message["content"] = formatted_content.strip()

        # print("`body` object after filtering:")
        # print(json.dumps(body, indent=4))
        # TODO Remove box formatting in the final response if it's too long.
        return body

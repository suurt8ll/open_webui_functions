"""
title: Smallthinker Filter
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.3.0
"""

from pydantic import BaseModel, Field
import json
from time import time


class Filter:
    class Valves(BaseModel):
        ENABLE_FORMATTING: bool = Field(default=True)
        SPLIT_KEYWORDS: str = Field(
            default="**Final Answer**",
            description="Comma-separated list of keywords to split the message content",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.start_time = 0

    def inlet(self, body: dict, **kwargs) -> dict:
        self.start_time = time()
        return body

    def outlet(self, body: dict, **kwargs) -> dict:
        print("`body` object before filtering:")
        print(json.dumps(body, indent=4))
        if self.valves.ENABLE_FORMATTING:
            split_keywords = [
                keyword.strip() for keyword in self.valves.SPLIT_KEYWORDS.split(",")
            ]
            print("Split keywords:", split_keywords)
            for message in body["messages"]:
                if message["role"] == "assistant":
                    content = message["content"]
                    split_index = -1
                    keyword_used = None
                    for keyword in split_keywords:
                        index = content.rfind(keyword)
                        if index > split_index:  # Modified to track the last index
                            split_index = index
                            keyword_used = keyword

                    if split_index != -1:
                        thoughts = content[:split_index].strip()
                        final_response = content[
                            split_index + len(keyword_used or "") :
                        ].strip()
                        formatted_content = ""
                        if thoughts:
                            formatted_content += (
                                "<details>\n<summary>💭 Thought for {:.0f} seconds</summary>\n".format(
                                    time() - self.start_time
                                )
                                + thoughts
                                + "\n</details>"
                            )
                        if final_response:
                            formatted_content += final_response
                        message["content"] = formatted_content.strip()

        print("`body` object after filtering:")
        print(json.dumps(body, indent=4))
        return body

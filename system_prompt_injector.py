"""
title: System Prompt Injector
description: Filter that will detect a system prompt from user input and inject it into the body object.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.3.0
"""

from pydantic import BaseModel
from typing import Optional


class Filter:
    # Valves: Configuration options for the filter
    class Valves(BaseModel):
        pass

    def __init__(self):
        # Initialize valves (optional configuration for the Filter)
        self.valves = self.Valves()

    def inlet(self, body: dict) -> dict:
        # This is where you manipulate user inputs.
        print(f"inlet called: {body}")
        return body

    def outlet(self, body: dict) -> None:
        # This is where you manipulate model outputs.
        print(f"outlet called: {body}")

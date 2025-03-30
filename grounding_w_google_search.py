"""
title: Grounding with Google Search
id: grounding_with_google_search
description: Filter function that gives Gemini models search ablities. Must be used with "Gemini Manifold google_genai" pipe function!
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.1.0
"""

import json


class Filter:

    def __init__(self):
        print("[grounding_w_google_search] Filter function has been initialized!")

    def inlet(self, body: dict) -> dict:
        print("INLET FILTER:")
        print(json.dumps(body, indent=2, default=str))
        return body

    def stream(self, event: dict) -> dict:
        print("STREAM FILTER:")
        print(json.dumps(event, indent=2, default=str))
        return event

    def outlet(self, body: dict) -> dict:
        print("OUTLET FILTER:")
        print(json.dumps(body, indent=2, default=str))
        return body

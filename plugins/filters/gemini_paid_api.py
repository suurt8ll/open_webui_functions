"""
title: Gemini Paid API
id: gemini_paid_api
description: Switches to paid Gemini Developer API
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.0.0
"""

from pydantic import BaseModel


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self) -> None:
        self.valves = self.Valves()
        # Makes the filter toggleable in the front-end.
        self.toggle = True
        # Icon from https://icon-sets.iconify.design/ic/outline-paid/
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0OCIgaGVpZ2h0PSI0OCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwczEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAybTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LThzOCAzLjU5IDggOHMtMy41OSA4LTggOG0uODktOC45Yy0xLjc4LS41OS0yLjY0LS45Ni0yLjY0LTEuOWMwLTEuMDIgMS4xMS0xLjM5IDEuODEtMS4zOWMxLjMxIDAgMS43OS45OSAxLjkgMS4zNGwxLjU4LS42N2MtLjE1LS40NC0uODItMS45MS0yLjY2LTIuMjNWNWgtMS43NXYxLjI2Yy0yLjYuNTYtMi42MiAyLjg1LTIuNjIgMi45NmMwIDIuMjcgMi4yNSAyLjkxIDMuMzUgMy4zMWMxLjU4LjU2IDIuMjggMS4wNyAyLjI4IDIuMDNjMCAxLjEzLTEuMDUgMS42MS0xLjk4IDEuNjFjLTEuODIgMC0yLjM0LTEuODctMi40LTIuMDlsLTEuNjYuNjdjLjYzIDIuMTkgMi4yOCAyLjc4IDMuMDIgMi45NlYxOWgxLjc1di0xLjI0Yy41Mi0uMDkgMy4wMi0uNTkgMy4wMi0zLjIyYy4wMS0xLjM5LS42LTIuNjEtMy0zLjQ0Ii8+PC9zdmc+"

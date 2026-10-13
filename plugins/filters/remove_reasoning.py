"""
title: Remove Reasoning
description: Filter function to remove reasoning from the response.
id: remove_reasoning
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 1.0.0
"""

import datetime
import inspect


class Filter:

    def __init__(self):
        self.toggle = True  # Makes the filter toggleable in the front-end.
        # Icon from https://icon-sets.iconify.design/mdi/brain/
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxZW0iIGhlaWdodD0iMWVtIiB2aWV3Qm94PSIwIDAgMjQgMjQiPgoJPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIgLz4KCTxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTIxLjMzIDEyLjkxYy4wOSAxLjU1LS42MiAzLjA0LTEuODkgMy45NWwuNzcgMS40OWMuMjMuNDUuMjYuOTguMDYgMS40NWMtLjE5LjQ3LS41OC44NC0xLjA2IDFsLS43OS4yNWExLjY5IDEuNjkgMCAwIDEtMS44Ni0uNTVMMTQuNDQgMThjLS44OS0uMTUtMS43My0uNTMtMi40NC0xLjFjLS41LjE1LTEgLjIzLTEuNS4yM2MtLjg4IDAtMS43Ni0uMjctMi41LS43OWMtLjUzLjE2LTEuMDcuMjMtMS42Mi4yMmMtLjc5LjAxLTEuNTctLjE1LTIuMy0uNDVhNC4xIDQuMSAwIDAgMS0yLjQzLTMuNjFjLS4wOC0uNzIuMDQtMS40NS4zNS0yLjExYy0uMjktLjc1LS4zMi0xLjU3LS4wNy0yLjMzQzIuMyA3LjExIDMgNi4zMiAzLjg3IDUuODJjLjU4LTEuNjkgMi4yMS0yLjgyIDQtMi43YzEuNi0xLjUgNC4wNS0xLjY2IDUuODMtLjM3Yy40Mi0uMTEuODYtLjE3IDEuMy0uMTdjMS4zNi0uMDMgMi42NS41NyAzLjUgMS42NGMyLjA0LjUzIDMuNSAyLjM1IDMuNTggNC40N2MuMDUgMS4xMS0uMjUgMi4yLS44NiAzLjEzYy4wNy4zNi4xMS43Mi4xMSAxLjA5bS01LTEuNDFjLjU3LjA3IDEuMDIuNSAxLjAyIDEuMDdhMSAxIDAgMCAxLTEgMWgtLjYzYy0uMzIuOS0uODggMS42OS0xLjYyIDIuMjljLjI1LjA5LjUxLjE0Ljc3LjIxYzUuMTMtLjA3IDQuNTMtMy4yIDQuNTMtMy4yNWEyLjU5IDIuNTkgMCAwIDAtMi42OS0yLjQ5YTEgMSAwIDAgMS0xLTFhMSAxIDAgMCAxIDEtMWMxLjIzLjAzIDIuNDEuNDkgMy4zMyAxLjNjLjA1LS4yOS4wOC0uNTkuMDgtLjg5Yy0uMDYtMS4yNC0uNjItMi4zMi0yLjg3LTIuNTNjLTEuMjUtMi45Ni00LjQtMS4zMi00LjQtLjRjLS4wMy4yMy4yMS43Mi4yNS43NWExIDEgMCAwIDEgMSAxYzAgLjU1LS40NSAxLTEgMWMtLjUzLS4wMi0xLjAzLS4yMi0xLjQzLS41NmMtLjQ4LjMxLTEuMDMuNS0xLjYuNTZjLS41Ny4wNS0xLjA0LS4zNS0xLjA3LS45YS45Ny45NyAwIDAgMSAuODgtMS4xYy4xNi0uMDIuOTQtLjE0Ljk0LS43N2MwLS42Ni4yNS0xLjI5LjY4LTEuNzljLS45Mi0uMjUtMS45MS4wOC0yLjkxIDEuMjlDNi43NSA1IDYgNS4yNSA1LjQ1IDcuMkM0LjUgNy42NyA0IDggMy43OCA5YzEuMDgtLjIyIDIuMTktLjEzIDMuMjIuMjVjLjUuMTkuNzguNzUuNTkgMS4yOWMtLjE5LjUyLS43Ny43OC0xLjI5LjU5Yy0uNzMtLjMyLTEuNTUtLjM0LTIuMy0uMDZjLS4zMi4yNy0uMzIuODMtLjMyIDEuMjdjMCAuNzQuMzcgMS40MyAxIDEuODNjLjUzLjI3IDEuMTIuNDEgMS43MS40cS0uMjI1LS4zOS0uMzktLjgxYTEuMDM4IDEuMDM4IDAgMCAxIDEuOTYtLjY4Yy40IDEuMTQgMS40MiAxLjkyIDIuNjIgMi4wNWMxLjM3LS4wNyAyLjU5LS44OCAzLjE5LTIuMTNjLjIzLTEuMzggMS4zNC0xLjUgMi41Ni0xLjVtMiA3LjQ3bC0uNjItMS4zbC0uNzEuMTZsMSAxLjI1em0tNC42NS04LjYxYTEgMSAwIDAgMC0uOTEtMS4wM2MtLjcxLS4wNC0xLjQuMi0xLjkzLjY3Yy0uNTcuNTgtLjg3IDEuMzgtLjg0IDIuMTlhMSAxIDAgMCAwIDEgMWMuNTcgMCAxLS40NSAxLTFjMC0uMjcuMDctLjU0LjIzLS43NmMuMTItLjEuMjctLjE1LjQzLS4xNWMuNTUuMDMgMS4wMi0uMzggMS4wMi0uOTIiIC8+Cjwvc3ZnPgo="
        self._log("Function has been initialized!")

    def _log(self, message: str):
        timestamp = datetime.datetime.now().isoformat()
        caller_name = inspect.stack()[1].function
        print(f"[{timestamp}] [{__name__}.{caller_name}] {message}")

    async def stream(
        self,
        event: dict,
    ) -> dict | None:
        # if event.choices[0].delta contains key reasoning_content, remove it
        if "choices" in event and len(event["choices"]) > 0:
            for choice in event["choices"]:
                if "delta" in choice and "reasoning_content" in choice["delta"]:
                    del choice["delta"]["reasoning_content"]
        return event

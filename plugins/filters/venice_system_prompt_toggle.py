"""
title: Disable Venice System Prompt
description: Filter function to turn off the Venice.ai's default system prompt.
id: disable_venice_system_prompt
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
        # Icon from https://icon-sets.iconify.design/material-symbols/comments-disabled-outline-rounded/
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxZW0iIGhlaWdodD0iMWVtIiB2aWV3Qm94PSIwIDAgMjQgMjQiPgoJPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIgLz4KCTxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMThxLS44MjUgMC0xLjQxMi0uNTg3VDIgMTZWNC44MjVMMS4zNzUgNC4ycS0uMy0uMy0uMy0uNzEzdC4zLS43MTJ0LjcxMy0uM3QuNzEyLjNsMTguNCAxOC40cS4zLjMuMy43dC0uMy43dC0uNzEyLjN0LS43MTMtLjNMMTUuMTc1IDE4ek0yMiA0djEzLjkyNXEwIC4zNS0uMy40NzV0LS41NS0uMTI1TDE4Ljg3NSAxNkgyMFY0SDdxLS41IDAtLjc1LS4zMTJUNiAzdC4yNS0uNjg3VDcgMmgxM3EuODI1IDAgMS40MTMuNTg4VDIyIDRNNCAxNmg5LjE3NWwtMi0ySDdxLS40MjUgMC0uNzEyLS4yODhUNiAxM3QuMjg4LS43MTJUNyAxMmgyLjE3NWwtMS0xSDdxLS40MjUgMC0uNzEyLS4yODhUNiAxMHQuMjg4LS43MTJUNyA5aC42MjV2MS40NUw0IDYuODI1em0xNC0zcTAgLjQyNS0uMjg4LjcxM1QxNyAxNHQtLjcxMi0uMjg4VDE2IDEzdC4yODgtLjcxMlQxNyAxMnQuNzEzLjI4OFQxOCAxM20tMS0yaC0yLjdxLS41IDAtLjc1LS4zMTJUMTMuMyAxMHQuMjUtLjY4N1QxNC4zIDlIMTdxLjQyNSAwIC43MTMuMjg4VDE4IDEwdC0uMjg4LjcxM1QxNyAxMW0wLTNoLTUuN3EtLjUgMC0uNzUtLjMxMlQxMC4zIDd0LjI1LS42ODdUMTEuMyA2SDE3cS40MjUgMCAuNzEzLjI4OFQxOCA3dC0uMjg4LjcxM1QxNyA4bS00LjEyNSAyIiAvPgo8L3N2Zz4K"
        self._log("Function has been initialized!")

    def _log(self, message: str):
        timestamp = datetime.datetime.now().isoformat()
        caller_name = inspect.stack()[1].function
        print(f"[{timestamp}] [{__name__}.{caller_name}] {message}")

    async def inlet(
        self,
        body: dict,
    ) -> dict:
        self._log("Disabling Venice.ai's default system prompt.")
        body["venice_parameters"] = {"include_venice_system_prompt": False}
        return body

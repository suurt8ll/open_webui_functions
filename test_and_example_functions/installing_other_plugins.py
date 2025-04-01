"""
title: Plugins within Plugins
id: installing_other_plugins
description: Stashed example of how to install plugins from inside another plugin.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.0
requirements:
"""

from pydantic import BaseModel, Field
import requests
from open_webui.models.functions import (
    FunctionForm,
    FunctionMeta,
    FunctionModel,
    Functions,
)
from open_webui.models.users import Users
from open_webui.utils.plugin import extract_frontmatter


SEARCH_FILTER_ID = "grounding_with_google_search"


class Pipe:

    class Valves(BaseModel):
        USE_GROUNDING_SEARCH: bool = Field(
            default=False,
            description="Whether to use Grounding with Google Search. For more info: https://ai.google.dev/gemini-api/docs/grounding",
        )

    def __init__(self):
        valves = Functions.get_function_valves_by_id("installing_other_plugins")
        self.valves = self.Valves(**(valves if valves else {}))

        search_filter = Functions.get_function_by_id(id=SEARCH_FILTER_ID)
        if not search_filter:
            print("Search filter helper is not installed!")
            search_filter = self._install_search_filter()
            if not search_filter:
                print("Search filter insertion failed.")
            else:
                print("Search filter installed successfully!")
                enabled_search_filter = Functions.update_function_by_id(
                    id=search_filter.id, updated={"is_active": True}
                )
                if not enabled_search_filter:
                    print("Search filter could not be enabled.")
                print("Search filter enabled.")
                # TODO: Activate the filter for models inside `ALLOWED_GROUNDING_MODELS`.
        else:
            # TODO: [refac] repeating logic here, find a better way.
            if self.valves.USE_GROUNDING_SEARCH and not search_filter.is_active:
                print("Search filter is installed but not active. Activating it.")
                enabled_search_filter = Functions.update_function_by_id(
                    id=search_filter.id, updated={"is_active": True}
                )
                if not enabled_search_filter:
                    print("Search filter could not be enabled.")
                print("Search filter enabled.")

    async def pipe(self, body: dict) -> str:
        return "Hello World!"

    def _install_search_filter(self) -> FunctionModel | None:
        # Fetch the code from my repo.
        # FIXME: Change to master branch before merging!
        raw_url = "https://raw.githubusercontent.com/suurt8ll/open_webui_functions/refs/heads/feat/grounding-search-filter-function/grounding_w_google_search.py"
        try:
            response = requests.get(raw_url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error fetching GitHub code: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error fetching GitHub code: {e}")
            return None
        github_code = response.text

        # Extract the metadata from the frontmatter.
        github_frontmatter = extract_frontmatter(github_code)
        function_id = github_frontmatter.get("id", __name__.split(".")[-1])

        # Construct FunctionForm payload.
        function_meta = FunctionMeta(
            description=github_frontmatter.get("description", ""),
            manifest=github_frontmatter,
        )
        function_form = FunctionForm(
            id=function_id,
            name=github_frontmatter.get("title", "Grounding with Google Search"),
            content=github_code,
            meta=function_meta,
        )

        # Use the first registered user's id, IDK if it matters who registeres the function.
        user_id = Users.get_first_user().id

        # Add the filter function to the backend.
        updated_function = Functions.insert_new_function(
            user_id=user_id, type="filter", form_data=function_form
        )

        return updated_function

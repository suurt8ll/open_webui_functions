"""
title: Dynamic Plugin Log Manager
id: dynamic_plugin_log_manager
description: Dynamically injects and manages Loguru handlers with custom formatting for Open WebUI plugins.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.0.0
"""

import copy
import json
import sys
from typing import Any, TYPE_CHECKING
import pydantic_core
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from loguru import Record


def _create_plugin_filter(target_module: str):
    def filter_func(record: "Record") -> bool:
        return record["name"] == target_module

    filter_func.target_module = target_module  # type: ignore
    return filter_func


class Pipe:
    class Valves(BaseModel):
        FUNCTIONS_LOG_LEVELS: str = Field(
            default='{}',
            description="JSON object mapping function IDs to Loguru log levels (e.g. TRACE, DEBUG, INFO, WARNING, ERROR).",
        )

    def __init__(self):
        self.valves = self.Valves()
        logger.success("Function has been initialized.")

    def _is_flat_dict(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        return not any(isinstance(value, (dict, list)) for value in data.values())

    def _truncate_long_strings(
        self, data: Any, max_len: int, truncation_marker: str, truncation_enabled: bool
    ) -> Any:
        if not truncation_enabled or max_len <= len(truncation_marker):
            if isinstance(data, (dict, list)):
                return copy.deepcopy(data)
            return data

        if isinstance(data, str):
            if len(data) > max_len:
                return data[: max_len - len(truncation_marker)] + truncation_marker
            return data
        elif isinstance(data, dict):
            return {
                k: self._truncate_long_strings(
                    v, max_len, truncation_marker, truncation_enabled
                )
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [
                self._truncate_long_strings(
                    item, max_len, truncation_marker, truncation_enabled
                )
                for item in data
            ]
        else:
            return data

    def plugin_stdout_format(self, record: "Record") -> str:
        LOG_OPTIONS_PREFIX = "_log_"
        TRUNCATION_ENABLED_KEY = f"{LOG_OPTIONS_PREFIX}truncation_enabled"
        MAX_LENGTH_KEY = f"{LOG_OPTIONS_PREFIX}max_length"
        TRUNCATION_MARKER_KEY = f"{LOG_OPTIONS_PREFIX}truncation_marker"
        DATA_KEY = "payload"

        original_extra = record["extra"]
        data_to_process = original_extra.get(DATA_KEY)

        serialized_data_json = ""
        if data_to_process is not None:
            try:
                serializable_data = pydantic_core.to_jsonable_python(
                    data_to_process, serialize_unknown=True
                )

                truncation_enabled = original_extra.get(TRUNCATION_ENABLED_KEY, True)
                max_length = original_extra.get(MAX_LENGTH_KEY, 256)
                truncation_marker = original_extra.get(TRUNCATION_MARKER_KEY, "[...]")

                if MAX_LENGTH_KEY in original_extra:
                    truncation_enabled = True

                truncated_data = self._truncate_long_strings(
                    serializable_data,
                    max_length,
                    truncation_marker,
                    truncation_enabled,
                )

                if self._is_flat_dict(truncated_data) and not isinstance(
                    truncated_data, list
                ):
                    json_string = json.dumps(
                        truncated_data, separators=(",", ":"), default=str
                    )
                    serialized_data_json = " - " + json_string
                else:
                    json_string = json.dumps(truncated_data, indent=2, default=str)
                    serialized_data_json = "\n" + json_string

            except (TypeError, ValueError) as e:
                serialized_data_json = f" - {{Serialization Error: {e}}}"
            except Exception as e:
                serialized_data_json = f" - {{Processing Error: {e}}}"

        record["extra"]["_plugin_serialized_data"] = serialized_data_json

        base_template = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
            "{extra[_plugin_serialized_data]}\n"
            "{exception}"
        )
        return base_template.rstrip()

    def _sync_log_handlers(self) -> None:
        if not self.valves.FUNCTIONS_LOG_LEVELS:
            return

        try:
            config = json.loads(self.valves.FUNCTIONS_LOG_LEVELS)
            if not isinstance(config, dict):
                logger.error(
                    "FUNCTIONS_LOG_LEVELS must be a JSON object mapping function IDs to log levels."
                )
                return
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse FUNCTIONS_LOG_LEVELS JSON: {e}")
            return

        target_levels: dict[str, str] = {}
        for func_id, level_name in config.items():
            if not isinstance(func_id, str) or not isinstance(level_name, str):
                continue

            clean_id = func_id.strip()
            if not clean_id:
                continue

            module_name = (
                clean_id if clean_id.startswith("function_") else f"function_{clean_id}"
            )
            level_name_upper = level_name.upper().strip()

            try:
                logger.level(level_name_upper)
                target_levels[module_name] = level_name_upper
            except ValueError:
                logger.error(
                    f"Invalid log level '{level_name}' configured for function '{func_id}'."
                )

        handlers: dict[int, Any] = logger._core.handlers  # type: ignore
        active_managed_modules: set[str] = set()

        for handler_id, handler in list(handlers.items()):
            existing_filter = handler._filter
            target_module = getattr(existing_filter, "target_module", None)

            if target_module is not None:
                desired_level = target_levels.get(target_module)

                if desired_level is not None:
                    desired_level_no = logger.level(desired_level).no
                    if handler.levelno == desired_level_no:
                        active_managed_modules.add(target_module)
                        continue

                try:
                    logger.remove(handler_id)
                    logger.info(
                        f"Removed Loguru handler {handler_id} for module '{target_module}'."
                    )
                except ValueError:
                    pass

        for module_name, level_name in target_levels.items():
            if module_name not in active_managed_modules:
                plugin_filter = _create_plugin_filter(module_name)
                logger.add(
                    sys.stdout,
                    level=level_name,
                    format=self.plugin_stdout_format,
                    filter=plugin_filter,
                )
                logger.success(
                    f"Added Loguru handler for module '{module_name}' with level '{level_name}'."
                )

    async def pipes(self) -> list[dict]:
        self._sync_log_handlers()
        return []

    async def pipe(self, body: dict) -> dict:
        return body

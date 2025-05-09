import pytest
from unittest.mock import (
    patch,
    MagicMock,
)  # No need for mock_open here for this specific problem
import sys  # Import sys to manipulate sys.modules

# --- Mock problematic Open WebUI modules BEFORE they are imported by your plugin ---
# We create MagicMock objects that will pretend to be these modules/classes.
# This prevents their actual code (including DB connections) from running.

mock_chats_module = MagicMock()
mock_files_module = MagicMock()
mock_functions_module = MagicMock()
mock_storage_module = MagicMock()

# If the imports in your plugin are like:
# from open_webui.models.chats import Chats
# Then 'Chats' is an attribute of the 'open_webui.models.chats' module.
# So, we mock the module and then assign a mock to the attribute.
mock_chats_module.Chats = MagicMock()

# from open_webui.models.files import FileForm, Files
mock_files_module.FileForm = MagicMock()
mock_files_module.Files = MagicMock()

# from open_webui.models.functions import Functions
mock_functions_module.Functions = MagicMock()  # This is the one used in __init__

# from open_webui.storage.provider import Storage
mock_storage_module.Storage = MagicMock()

# Add these mocks to sys.modules so that when 'gemini_manifold.py'
# tries to import them, it gets our mocks instead of the real modules.
sys.modules["open_webui.models.chats"] = mock_chats_module
sys.modules["open_webui.models.files"] = mock_files_module
sys.modules["open_webui.models.functions"] = mock_functions_module
sys.modules["open_webui.storage.provider"] = mock_storage_module

# --- Now, the import of your plugin should use the mocks ---
from plugins.pipes.gemini_manifold import Pipe, types as gemini_types
from pydantic import BaseModel


# Your fixture and test function from before
@pytest.fixture
def mock_pipe_valves_data():
    return {
        "GEMINI_API_KEY": "test_default_api_key_from_valves",
        "REQUIRE_USER_API_KEY": False,
        "GEMINI_API_BASE_URL": "https://test.googleapis.com",
        "MODEL_WHITELIST": "*",
        "MODEL_BLACKLIST": None,
        "CACHE_MODELS": True,
        "THINKING_BUDGET": 8192,
        "USE_FILES_API": True,
        "THINKING_MODEL_PATTERN": r"thinking|gemini-2.5",
        "EMIT_INTERVAL": 1,
        "EMIT_STATUS_UPDATES": False,
        "LOG_LEVEL": "INFO",
    }


def test_pipe_initialization_with_api_key(mock_pipe_valves_data):
    mock_gemini_client_instance = MagicMock()

    # The Functions class is used in __init__
    # We've already put a mock for the 'open_webui.models.functions' module in sys.modules,
    # and that mock has a 'Functions' attribute which is also a MagicMock.
    # So, when Pipe.__init__ does:
    #   valves = Functions.get_function_valves_by_id("gemini_manifold_google_genai")
    # it will be calling mock_functions_module.Functions.get_function_valves_by_id(...)

    # Configure the mock for Functions.get_function_valves_by_id
    # This is now mock_functions_module.Functions because that's what
    # 'from open_webui.models.functions import Functions' will resolve to.
    mock_functions_module.Functions.get_function_valves_by_id.return_value = (
        mock_pipe_valves_data
    )

    # We still need to patch genai.Client and the internal _add_log_handler
    with patch(
        "plugins.pipes.gemini_manifold.genai.Client",
        return_value=mock_gemini_client_instance,
    ) as MockedGenAIClientConstructor, patch.object(
        Pipe, "_add_log_handler", MagicMock()
    ) as mock_internal_add_log_handler, patch(
        "sys.stdout", MagicMock()
    ):  # Suppresses print/log output to stdout

        # ---- Instantiate the Pipe class ----
        pipe_instance = Pipe()
        # ------------------------------------

        # ---- Assertions ----
        mock_functions_module.Functions.get_function_valves_by_id.assert_called_once_with(
            "gemini_manifold_google_genai"
        )

        assert isinstance(pipe_instance.valves, Pipe.Valves)
        assert pipe_instance.valves.GEMINI_API_KEY == "test_default_api_key_from_valves"
        assert pipe_instance.valves.LOG_LEVEL == "INFO"
        assert pipe_instance.log_level == "INFO"
        mock_internal_add_log_handler.assert_called_once()
        MockedGenAIClientConstructor.assert_called_once_with(
            api_key="test_default_api_key_from_valves",
            http_options=gemini_types.HttpOptions(
                base_url="https://test.googleapis.com"
            ),
        )
        assert pipe_instance.clients["default"] == mock_gemini_client_instance
        assert pipe_instance.models == []
        assert pipe_instance.last_whitelist == "*"
        assert pipe_instance.last_blacklist is None


# It's good practice to clean up sys.modules if you modify it,
# though for simple test runs it might not always be strictly necessary.
# A fixture with a finalizer or teardown_module can do this.
def teardown_module(module):
    """Cleans up sys.modules after tests in this file are done."""
    del sys.modules["open_webui.models.chats"]
    del sys.modules["open_webui.models.files"]
    del sys.modules["open_webui.models.functions"]
    del sys.modules["open_webui.storage.provider"]

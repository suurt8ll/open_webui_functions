from typing import cast
from aiocache.base import BaseCache
import pytest
import pytest_asyncio
from unittest.mock import (
    patch,
    MagicMock,
    AsyncMock,
    call,
)
import sys

# --- Mock problematic Open WebUI modules BEFORE they are imported by your plugin ---
mock_chats_module = MagicMock()
mock_files_module = MagicMock()
mock_functions_module = MagicMock()
mock_storage_module = MagicMock()

mock_chats_module.Chats = MagicMock()
mock_files_module.FileForm = MagicMock()
mock_files_module.Files = MagicMock()
mock_functions_module.Functions = MagicMock()
mock_storage_module.Storage = MagicMock()

sys.modules["open_webui.models.chats"] = mock_chats_module
sys.modules["open_webui.models.files"] = mock_files_module
sys.modules["open_webui.models.functions"] = mock_functions_module
sys.modules["open_webui.storage.provider"] = mock_storage_module

# --- Now, the import of your plugin should use the mocks ---
from plugins.pipes.gemini_manifold import (
    Pipe,
    types as gemini_types,
)  # gemini_types is google.genai.types


# region Fixtures
@pytest.fixture
def mock_pipe_valves_data():
    """
    Fixture to provide a base set of valves data.
    Updated to include all fields from Pipe.Valves for robust Pydantic initialization.
    """
    return {
        "GEMINI_API_KEY": "test_default_api_key_from_valves",
        "USER_MUST_PROVIDE_AUTH_CONFIG": False,
        "AUTH_WHITELIST": None,
        "GEMINI_API_BASE_URL": "https://test.googleapis.com",
        "USE_VERTEX_AI": False,
        "VERTEX_PROJECT": None,
        "VERTEX_LOCATION": "global",
        "MODEL_WHITELIST": "*",
        "MODEL_BLACKLIST": None,
        "CACHE_MODELS": True,
        "THINKING_BUDGET": 8192,
        "USE_FILES_API": True,
        "THINKING_MODEL_PATTERN": r"thinking|gemini-2.5",
        "EMIT_INTERVAL": 1,
        "EMIT_STATUS_UPDATES": False,
        "LOG_LEVEL": "INFO",
        "ENABLE_URL_CONTEXT_TOOL": False,
    }


@pytest_asyncio.fixture
async def pipe_instance_fixture(mock_pipe_valves_data):
    """
    Helper fixture to setup a Pipe instance with mocked genai.Client constructor
    and yields both the pipe instance and the mock constructor.
    """
    mock_gemini_client_actual_instance = MagicMock()

    with patch(
        "plugins.pipes.gemini_manifold.genai.Client",
        return_value=mock_gemini_client_actual_instance,  # The constructor returns this
    ) as MockedGenAIClientConstructor, patch.object(  # MockedGenAIClientConstructor is the mock of genai.Client class
        Pipe, "_add_log_handler", MagicMock()
    ), patch(
        "sys.stdout", MagicMock()
    ):
        pipe = Pipe()
        # Initialize with base data from mock_pipe_valves_data
        pipe.valves = Pipe.Valves(**mock_pipe_valves_data)
        # Yield both the pipe instance and the mock for genai.Client constructor
        yield pipe, MockedGenAIClientConstructor

    # Teardown: Clear caches to ensure clean state for subsequent tests
    Pipe._get_or_create_genai_client.cache_clear()
    # Check if _get_genai_models has a cache attribute before trying to clear it
    if hasattr(pipe._get_genai_models, "cache"):
        cache_instance = getattr(pipe._get_genai_models, "cache")
        if cache_instance:
            await cast(BaseCache, cache_instance).clear()


# endregion Fixtures


# region Test _get_or_create_genai_client
def test_pipe_initialization_with_api_key(mock_pipe_valves_data):
    """
    Tests basic Pipe initialization and client creation when an API key is provided.
    """
    mock_gemini_client_instance = MagicMock()

    with patch(
        "plugins.pipes.gemini_manifold.genai.Client",
        return_value=mock_gemini_client_instance,
    ) as MockedGenAIClientConstructor, patch.object(
        Pipe, "_add_log_handler", MagicMock()
    ), patch(
        "sys.stdout", MagicMock()
    ):
        try:
            pipe_instance = Pipe()
            pipe_instance.valves = Pipe.Valves(**mock_pipe_valves_data)

            assert isinstance(pipe_instance.valves, Pipe.Valves)
            assert (
                pipe_instance.valves.GEMINI_API_KEY
                == "test_default_api_key_from_valves"
            )

            # Trigger client creation
            pipe_instance._get_user_client(
                pipe_instance.valves, "test_user_email@example.com"
            )

            MockedGenAIClientConstructor.assert_called_once_with(
                api_key="test_default_api_key_from_valves",
                http_options=gemini_types.HttpOptions(
                    base_url="https://test.googleapis.com"
                ),
            )
        finally:
            Pipe._get_or_create_genai_client.cache_clear()


def test_get_user_client_no_auth_provided_raises_error(mock_pipe_valves_data):
    """
    Tests that genai.Client is NOT called when neither GEMINI_API_KEY
    nor VERTEX_PROJECT is provided and an error is raised.
    """
    # Configure valves to have no API key and no Vertex AI project
    mock_pipe_valves_data["GEMINI_API_KEY"] = None
    mock_pipe_valves_data["USE_VERTEX_AI"] = False  # Explicitly set to False
    mock_pipe_valves_data["VERTEX_PROJECT"] = None
    # Ensure USER_MUST_PROVIDE_AUTH_CONFIG is False so the check happens inside _get_or_create_genai_client
    mock_pipe_valves_data["USER_MUST_PROVIDE_AUTH_CONFIG"] = False
    mock_pipe_valves_data["AUTH_WHITELIST"] = None  # Ensure user is not whitelisted

    mock_gemini_client_instance = MagicMock()

    with patch(
        "plugins.pipes.gemini_manifold.genai.Client",
        return_value=mock_gemini_client_instance,
    ) as MockedGenAIClientConstructor, patch.object(
        Pipe, "_add_log_handler", MagicMock()
    ), patch(
        "sys.stdout", MagicMock()  # Suppress log output during test
    ):
        pipe_instance = Pipe()
        pipe_instance.valves = Pipe.Valves(**mock_pipe_valves_data)

        # We expect a ValueError from _get_user_client
        with pytest.raises(ValueError):
            pipe_instance._get_user_client(
                pipe_instance.valves, "test_user_email@example.com"
            )

        # Assert that genai.Client was NOT called
        MockedGenAIClientConstructor.assert_not_called()

        # Clean up cache for _get_or_create_genai_client
        Pipe._get_or_create_genai_client.cache_clear()


def test_client_creation_uses_gemini_api_when_configured(pipe_instance_fixture):
    """
    Tests that Gemini Developer API client is created when GEMINI_API_KEY is provided
    and USE_VERTEX_AI is False.
    Assumes USER_MUST_PROVIDE_AUTH_CONFIG is False.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()  # Ensure fresh call for this test's params

    # Configure valves for this specific scenario
    pipe.valves.GEMINI_API_KEY = "test_gemini_key_scenario1"
    pipe.valves.USE_VERTEX_AI = False
    pipe.valves.VERTEX_PROJECT = None  # Should be ignored
    pipe.valves.GEMINI_API_BASE_URL = "https://gemini.specific.example.com"
    # USER_MUST_PROVIDE_AUTH_CONFIG is False from mock_pipe_valves_data

    # Act
    client = pipe._get_user_client(pipe.valves, "testuser1@example.com")

    # Assert
    MockedGenAIClientConstructor.assert_called_once_with(
        api_key="test_gemini_key_scenario1",
        http_options=gemini_types.HttpOptions(
            base_url="https://gemini.specific.example.com"
        ),
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()  # Cleanup


def test_client_creation_uses_vertex_ai_when_configured(pipe_instance_fixture):
    """
    Tests that Vertex AI client is created when USE_VERTEX_AI is True
    and VERTEX_PROJECT is provided.
    Assumes USER_MUST_PROVIDE_AUTH_CONFIG is False.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Configure valves for this specific scenario
    pipe.valves.GEMINI_API_KEY = "should_be_ignored_key"  # Should be ignored for Vertex
    pipe.valves.USE_VERTEX_AI = True
    pipe.valves.VERTEX_PROJECT = "test_vertex_project_id"
    pipe.valves.VERTEX_LOCATION = "europe-west4"
    # USER_MUST_PROVIDE_AUTH_CONFIG is False from mock_pipe_valves_data

    # Act
    client = pipe._get_user_client(pipe.valves, "testuser2@example.com")

    # Assert
    MockedGenAIClientConstructor.assert_called_once_with(
        vertexai=True, project="test_vertex_project_id", location="europe-west4"
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()


def test_client_creation_falls_back_to_gemini_api_with_warning(pipe_instance_fixture):
    """
    Tests that Gemini Developer API client is used with a warning when USE_VERTEX_AI is True,
    VERTEX_PROJECT is not provided, but GEMINI_API_KEY is available.
    Assumes USER_MUST_PROVIDE_AUTH_CONFIG is False.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Configure valves for this specific scenario
    pipe.valves.GEMINI_API_KEY = "test_fallback_gemini_key"
    pipe.valves.USE_VERTEX_AI = True  # Attempt to use Vertex
    pipe.valves.VERTEX_PROJECT = None  # But no project ID
    pipe.valves.GEMINI_API_BASE_URL = "https://fallback.gemini.example.com"
    # USER_MUST_PROVIDE_AUTH_CONFIG is False from mock_pipe_valves_data

    # Patch the logger used within the method being tested
    with patch("plugins.pipes.gemini_manifold.log.warning") as mock_log_warning:
        # Act
        client = pipe._get_user_client(pipe.valves, "testuser3@example.com")

        # Assert log warning
        mock_log_warning.assert_called_once_with(
            "Vertex AI is enabled but no project is set. Using Gemini Developer API."
        )

    # Assert client creation (should be Gemini API)
    MockedGenAIClientConstructor.assert_called_once_with(
        api_key="test_fallback_gemini_key",
        http_options=gemini_types.HttpOptions(
            base_url="https://fallback.gemini.example.com"
        ),
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()


# endregion Test _get_or_create_genai_client


# region Test USER_MUST_PROVIDE_AUTH_CONFIG=True scenarios
USER_EMAIL_UNPRIVILEGED = "unprivileged_user@example.com"
ADMIN_API_KEY_DEFAULT = "admin_api_key_should_not_be_used"
ADMIN_VERTEX_PROJECT_DEFAULT = "admin_vertex_project_should_not_be_used"
ADMIN_BASE_URL_DEFAULT = "https://admin.default.api.com"
ADMIN_VERTEX_LOCATION_DEFAULT = "admin-default-location"


def test_user_must_auth_no_user_key_provided_errors(pipe_instance_fixture):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True.
    User does NOT provide GEMINI_API_KEY in UserValves.
    Expected: ValueError, Client not called.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Setup admin/default valves
    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = None
    pipe.valves.GEMINI_API_KEY = ADMIN_API_KEY_DEFAULT
    pipe.valves.VERTEX_PROJECT = ADMIN_VERTEX_PROJECT_DEFAULT

    # User provides no API key
    user_valves_instance = Pipe.UserValves(GEMINI_API_KEY=None)
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_UNPRIVILEGED
    )

    with pytest.raises(ValueError) as excinfo:
        pipe._get_user_client(merged_valves, USER_EMAIL_UNPRIVILEGED)

    assert "User must provide their own authentication configuration" in str(
        excinfo.value
    )
    MockedGenAIClientConstructor.assert_not_called()
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_must_auth_user_provides_gemini_key_uses_user_creds(pipe_instance_fixture):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True.
    User provides GEMINI_API_KEY and USE_VERTEX_AI=False in UserValves.
    Expected: Gemini Developer API client with user's credentials.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Setup admin/default valves
    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = None
    pipe.valves.GEMINI_API_KEY = ADMIN_API_KEY_DEFAULT  # Admin key
    pipe.valves.VERTEX_PROJECT = ADMIN_VERTEX_PROJECT_DEFAULT  # Admin project
    pipe.valves.GEMINI_API_BASE_URL = ADMIN_BASE_URL_DEFAULT  # Admin base URL

    # User provides their own Gemini key and base URL
    user_api_key = "user_specific_gemini_key"
    user_base_url = "https://user.specific.api.com"
    user_valves_instance = Pipe.UserValves(
        GEMINI_API_KEY=user_api_key,
        USE_VERTEX_AI=False,
        GEMINI_API_BASE_URL=user_base_url,
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_UNPRIVILEGED
    )

    # Act
    client = pipe._get_user_client(merged_valves, USER_EMAIL_UNPRIVILEGED)

    # Assert
    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=user_api_key,  # Should use user's API key
        http_options=gemini_types.HttpOptions(
            base_url=user_base_url
        ),  # Should use user's base URL
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_must_auth_user_tries_vertex_no_user_gemini_key_errors(
    pipe_instance_fixture,
):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True.
    User attempts to use Vertex (USE_VERTEX_AI=True, VERTEX_PROJECT set in UserValves)
    but does NOT provide GEMINI_API_KEY in UserValves.
    Expected: ValueError because user's Vertex usage is denied, and they lack a fallback Gemini key. Client not called.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Setup admin/default valves
    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = None
    pipe.valves.GEMINI_API_KEY = ADMIN_API_KEY_DEFAULT
    pipe.valves.VERTEX_PROJECT = ADMIN_VERTEX_PROJECT_DEFAULT

    # User attempts Vertex but provides no Gemini API key
    user_valves_instance = Pipe.UserValves(
        USE_VERTEX_AI=True,
        VERTEX_PROJECT="user_tries_this_project",
        GEMINI_API_KEY=None,  # Crucially, no user Gemini key
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_UNPRIVILEGED
    )

    # Merged valves should have forced USE_VERTEX_AI to False and GEMINI_API_KEY to user's (which is None)
    assert merged_valves.USE_VERTEX_AI is False
    assert merged_valves.GEMINI_API_KEY is None

    with pytest.raises(ValueError) as excinfo:
        pipe._get_user_client(merged_valves, USER_EMAIL_UNPRIVILEGED)

    assert "User must provide their own authentication configuration" in str(
        excinfo.value
    )
    MockedGenAIClientConstructor.assert_not_called()
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_must_auth_user_tries_vertex_with_user_gemini_key_falls_back_to_user_gemini(
    pipe_instance_fixture,
):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True.
    User attempts to use Vertex (USE_VERTEX_AI=True in UserValves)
    AND provides GEMINI_API_KEY in UserValves.
    Expected: Falls back to Gemini Developer API using user's Gemini key, as user Vertex usage is denied.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Setup admin/default valves
    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = None
    pipe.valves.GEMINI_API_KEY = ADMIN_API_KEY_DEFAULT
    pipe.valves.VERTEX_PROJECT = ADMIN_VERTEX_PROJECT_DEFAULT
    pipe.valves.GEMINI_API_BASE_URL = ADMIN_BASE_URL_DEFAULT  # Admin base URL

    # User attempts Vertex but also provides a Gemini key and their own base URL
    user_fallback_api_key = "user_fallback_gemini_key"
    user_fallback_base_url = "https://user.fallback.api.com"
    user_valves_instance = Pipe.UserValves(
        USE_VERTEX_AI=True,  # User attempts this
        VERTEX_PROJECT="user_tries_this_project_too",  # User attempts this
        GEMINI_API_KEY=user_fallback_api_key,  # User provides this
        GEMINI_API_BASE_URL=user_fallback_base_url,  # User provides this
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_UNPRIVILEGED
    )

    # Merged valves should have forced USE_VERTEX_AI to False, VERTEX_PROJECT to None,
    # and GEMINI_API_KEY to user's.
    assert merged_valves.USE_VERTEX_AI is False
    assert merged_valves.VERTEX_PROJECT is None
    assert merged_valves.GEMINI_API_KEY == user_fallback_api_key
    assert merged_valves.GEMINI_API_BASE_URL == user_fallback_base_url

    # Act
    client = pipe._get_user_client(merged_valves, USER_EMAIL_UNPRIVILEGED)

    # Assert: Client should be Gemini Developer API with user's fallback key and base URL
    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=user_fallback_api_key,
        http_options=gemini_types.HttpOptions(base_url=user_fallback_base_url),
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()


# endregion Test USER_MUST_PROVIDE_AUTH_CONFIG=True scenarios


# region Test USER_MUST_PROVIDE_AUTH_CONFIG=True with whitelisted user
# You might already have these or similar constants
USER_EMAIL_WHITELISTED = "whitelisted_user@example.com"
ADMIN_GEMINI_KEY = "admin_default_gemini_key"
ADMIN_GEMINI_BASE_URL = "https://admin.default.gemini.api.com"
ADMIN_VERTEX_PROJECT = "admin_default_vertex_project"
ADMIN_VERTEX_LOCATION = "admin_default_vertex_location"


def test_whitelist_user_no_uservalves_uses_admin_gemini_config(pipe_instance_fixture):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True, user is whitelisted.
    User provides NO UserValves. Admin configured for Gemini API.
    Expected: Uses admin's Gemini API key and base URL.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Setup admin/default valves
    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = USER_EMAIL_WHITELISTED
    pipe.valves.GEMINI_API_KEY = ADMIN_GEMINI_KEY
    pipe.valves.GEMINI_API_BASE_URL = ADMIN_GEMINI_BASE_URL
    pipe.valves.USE_VERTEX_AI = False
    pipe.valves.VERTEX_PROJECT = None  # Ensure no Vertex config from admin

    # User provides no specific valves
    user_valves_instance = Pipe.UserValves()  # Empty or default UserValves
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_WHITELISTED
    )

    # Act
    client = pipe._get_user_client(merged_valves, USER_EMAIL_WHITELISTED)

    # Assert: Should use admin's Gemini config
    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=ADMIN_GEMINI_KEY,
        http_options=gemini_types.HttpOptions(base_url=ADMIN_GEMINI_BASE_URL),
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()


def test_whitelist_user_no_uservalves_uses_admin_vertex_config(pipe_instance_fixture):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True, user is whitelisted.
    User provides NO UserValves. Admin configured for Vertex AI.
    Expected: Uses admin's Vertex project and location.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Setup admin/default valves
    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = USER_EMAIL_WHITELISTED
    pipe.valves.GEMINI_API_KEY = None  # Ensure no Gemini config from admin
    pipe.valves.USE_VERTEX_AI = True
    pipe.valves.VERTEX_PROJECT = ADMIN_VERTEX_PROJECT
    pipe.valves.VERTEX_LOCATION = ADMIN_VERTEX_LOCATION

    # User provides no specific valves
    user_valves_instance = Pipe.UserValves()
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_WHITELISTED
    )

    # Act
    client = pipe._get_user_client(merged_valves, USER_EMAIL_WHITELISTED)

    # Assert: Should use admin's Vertex config
    MockedGenAIClientConstructor.assert_called_once_with(
        vertexai=True, project=ADMIN_VERTEX_PROJECT, location=ADMIN_VERTEX_LOCATION
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()


def test_whitelist_user_provides_own_gemini_key_overrides_admin(pipe_instance_fixture):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True, user is whitelisted.
    User provides their OWN Gemini API key in UserValves.
    Expected: Uses user's Gemini API key, ignoring admin's.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Setup admin/default valves (admin also has a Gemini key)
    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = USER_EMAIL_WHITELISTED
    pipe.valves.GEMINI_API_KEY = ADMIN_GEMINI_KEY  # Admin's key, should be overridden
    pipe.valves.GEMINI_API_BASE_URL = (
        ADMIN_GEMINI_BASE_URL  # Admin's base, should be overridden
    )
    pipe.valves.USE_VERTEX_AI = False

    # User provides their own Gemini key and base URL
    user_specific_key = "user_whitelisted_gemini_key"
    user_specific_base_url = "https://user.whitelisted.gemini.api.com"
    user_valves_instance = Pipe.UserValves(
        GEMINI_API_KEY=user_specific_key,
        GEMINI_API_BASE_URL=user_specific_base_url,
        USE_VERTEX_AI=False,  # Explicitly user wants Gemini
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_WHITELISTED
    )

    # Act
    client = pipe._get_user_client(merged_valves, USER_EMAIL_WHITELISTED)

    # Assert: Should use user's provided Gemini config
    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=user_specific_key,
        http_options=gemini_types.HttpOptions(base_url=user_specific_base_url),
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()


def test_whitelist_user_provides_own_vertex_config_overrides_admin(
    pipe_instance_fixture,
):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True, user is whitelisted.
    User provides their OWN Vertex project/location in UserValves.
    Expected: Uses user's Vertex config, ignoring admin's.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Setup admin/default valves (admin also has Vertex config)
    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = USER_EMAIL_WHITELISTED
    pipe.valves.USE_VERTEX_AI = True  # Admin default
    pipe.valves.VERTEX_PROJECT = (
        ADMIN_VERTEX_PROJECT  # Admin's project, should be overridden
    )
    pipe.valves.VERTEX_LOCATION = (
        ADMIN_VERTEX_LOCATION  # Admin's location, should be overridden
    )

    # User provides their own Vertex config
    user_specific_project = "user_whitelisted_vertex_project"
    user_specific_location = "user-whitelisted-vertex-location"
    user_valves_instance = Pipe.UserValves(
        USE_VERTEX_AI=True,  # User explicitly wants Vertex
        VERTEX_PROJECT=user_specific_project,
        VERTEX_LOCATION=user_specific_location,
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_WHITELISTED
    )

    # Act
    client = pipe._get_user_client(merged_valves, USER_EMAIL_WHITELISTED)

    # Assert: Should use user's provided Vertex config
    MockedGenAIClientConstructor.assert_called_once_with(
        vertexai=True, project=user_specific_project, location=user_specific_location
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()


# endregion Test USER_MUST_PROVIDE_AUTH_CONFIG=True with whitelisted user


# region Test user's ability to override admin's settings
# You might already have these or similar constants
USER_EMAIL_REGULAR = (
    "regular_user@example.com"  # A non-admin, non-whitelisted (for these tests) user
)
ADMIN_GEMINI_KEY = "admin_default_gemini_key"
ADMIN_GEMINI_BASE_URL = "https://admin.default.gemini.api.com"
ADMIN_VERTEX_PROJECT = "admin_default_vertex_project"
ADMIN_VERTEX_LOCATION = "admin_default_vertex_location"

USER_GEMINI_KEY = "user_specific_gemini_key"
USER_GEMINI_BASE_URL = "https://user.specific.gemini.api.com"
USER_VERTEX_PROJECT = "user_specific_vertex_project"
USER_VERTEX_LOCATION = "user_specific_vertex_location"


def test_user_opts_out_of_admin_vertex_to_user_gemini(pipe_instance_fixture):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=False. Admin uses Vertex.
    User provides UserValves to opt-out (USE_VERTEX_AI=False) and use their own Gemini key.
    Expected: Gemini Developer API client with user's key.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Setup admin/default valves: Admin is configured for Vertex
    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = False
    pipe.valves.AUTH_WHITELIST = None  # Not relevant here
    pipe.valves.USE_VERTEX_AI = True
    pipe.valves.VERTEX_PROJECT = ADMIN_VERTEX_PROJECT
    pipe.valves.VERTEX_LOCATION = ADMIN_VERTEX_LOCATION
    pipe.valves.GEMINI_API_KEY = ADMIN_GEMINI_KEY  # Admin might also have a Gemini key

    # User provides UserValves to opt-out of Vertex and use their own Gemini key
    user_valves_instance = Pipe.UserValves(
        USE_VERTEX_AI=False,  # User explicitly wants Gemini
        GEMINI_API_KEY=USER_GEMINI_KEY,
        GEMINI_API_BASE_URL=USER_GEMINI_BASE_URL,
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_REGULAR
    )

    # Act
    client = pipe._get_user_client(merged_valves, USER_EMAIL_REGULAR)

    # Assert: Should use user's Gemini config
    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=USER_GEMINI_KEY,
        http_options=gemini_types.HttpOptions(base_url=USER_GEMINI_BASE_URL),
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_opts_in_to_vertex_from_admin_gemini(pipe_instance_fixture):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=False. Admin uses Gemini API.
    User provides UserValves to opt-in (USE_VERTEX_AI=True) and use their own Vertex project.
    Expected: Vertex AI client with user's project.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Setup admin/default valves: Admin is configured for Gemini
    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = False
    pipe.valves.AUTH_WHITELIST = None
    pipe.valves.GEMINI_API_KEY = ADMIN_GEMINI_KEY
    pipe.valves.GEMINI_API_BASE_URL = ADMIN_GEMINI_BASE_URL
    pipe.valves.USE_VERTEX_AI = False
    pipe.valves.VERTEX_PROJECT = None  # Admin has no Vertex project

    # User provides UserValves to opt-in to Vertex with their own project
    user_valves_instance = Pipe.UserValves(
        USE_VERTEX_AI=True,  # User explicitly wants Vertex
        VERTEX_PROJECT=USER_VERTEX_PROJECT,
        VERTEX_LOCATION=USER_VERTEX_LOCATION,
        # User might or might not provide a GEMINI_API_KEY, shouldn't matter for Vertex
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_REGULAR
    )

    # Act
    client = pipe._get_user_client(merged_valves, USER_EMAIL_REGULAR)

    # Assert: Should use user's Vertex config
    MockedGenAIClientConstructor.assert_called_once_with(
        vertexai=True, project=USER_VERTEX_PROJECT, location=USER_VERTEX_LOCATION
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_overrides_admin_vertex_project_location(pipe_instance_fixture):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=False. Admin uses Vertex.
    User provides UserValves with different Vertex project and location.
    Expected: Vertex AI client with user's specified project and location.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Setup admin/default valves: Admin is configured for Vertex
    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = False
    pipe.valves.AUTH_WHITELIST = None
    pipe.valves.USE_VERTEX_AI = True
    pipe.valves.VERTEX_PROJECT = ADMIN_VERTEX_PROJECT  # Admin's project
    pipe.valves.VERTEX_LOCATION = ADMIN_VERTEX_LOCATION  # Admin's location

    # User provides UserValves to override Vertex project and location
    # USE_VERTEX_AI can be None in UserValves if admin already set it to True,
    # or True to be explicit.
    user_valves_instance = Pipe.UserValves(
        USE_VERTEX_AI=True,  # Or None, as admin default is True
        VERTEX_PROJECT=USER_VERTEX_PROJECT,  # User's override
        VERTEX_LOCATION=USER_VERTEX_LOCATION,  # User's override
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_REGULAR
    )

    # Act
    client = pipe._get_user_client(merged_valves, USER_EMAIL_REGULAR)

    # Assert: Should use user's overridden Vertex config
    MockedGenAIClientConstructor.assert_called_once_with(
        vertexai=True, project=USER_VERTEX_PROJECT, location=USER_VERTEX_LOCATION
    )
    assert client is MockedGenAIClientConstructor.return_value
    Pipe._get_or_create_genai_client.cache_clear()


# endregion Test user's ability to override admin's settings


# region Test _get_genai_models
# TODO: Add tests for _get_genai_models
# endregion Test _get_genai_models


# region Test _genai_contents_from_messages
@pytest.mark.asyncio
async def test_genai_contents_from_messages_simple_user_text(pipe_instance_fixture):
    """
    Tests conversion of a simple user text message into genai.types.Content.
    """
    pipe_instance, MockedGenAIClientConstructor = pipe_instance_fixture

    messages_body = [{"role": "user", "content": "Hello!"}]
    messages_db = None
    upload_documents = True
    mock_event_emitter = AsyncMock()

    with patch(
        "plugins.pipes.gemini_manifold.types.Part.from_text"
    ) as mock_part_from_text:
        mock_part_from_text.return_value = MagicMock(spec=gemini_types.Part)

        contents = await pipe_instance._genai_contents_from_messages(
            messages_body,
            messages_db,
            upload_documents,
            mock_event_emitter,
        )

        mock_part_from_text.assert_called_once_with(text="Hello!")
        assert len(contents) == 1
        content_item = contents[0]
        assert content_item.role == "user"
        assert len(content_item.parts) == 1
        assert content_item.parts[0] == mock_part_from_text.return_value
        mock_event_emitter.assert_not_called()


@pytest.mark.asyncio
async def test_genai_contents_from_messages_youtube_link_mixed_with_text(
    pipe_instance_fixture,
):
    pipe_instance, MockedGenAIClientConstructor = pipe_instance_fixture

    # Arrange: Inputs
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    text_before_raw = "Look at this: "  # Has trailing space
    text_after_raw = " it's great!"  # Has leading space

    # Expected stripped text for Part.from_text calls
    text_before_stripped = text_before_raw.strip()
    text_after_stripped = text_after_raw.strip()

    user_content_string = f"{text_before_raw}{youtube_url}{text_after_raw}"
    messages_body = [{"role": "user", "content": user_content_string}]
    messages_db = None
    upload_documents = (
        True  # Does not affect this specific test path but set for consistency
    )
    mock_event_emitter = AsyncMock()

    # Mock types.Part.from_text to control its return values for distinct text segments
    mock_text_part_before_obj = MagicMock(spec=gemini_types.Part, name="TextPartBefore")
    mock_text_part_after_obj = MagicMock(spec=gemini_types.Part, name="TextPartAfter")

    # This patch targets 'types.Part.from_text' as used within the gemini_manifold module.
    with patch(
        "plugins.pipes.gemini_manifold.types.Part.from_text"
    ) as mock_part_from_text:
        # Configure side_effect to return specific mocks for expected text segments.
        # This helps verify that from_text is called with the correct, segmented text.
        def from_text_side_effect(text):
            if text == text_before_stripped:
                return mock_text_part_before_obj
            elif text == text_after_stripped:
                return mock_text_part_after_obj
            # If called with unexpected text (e.g., the full string containing the URL),
            # return a generic mock or raise an error to make test failure clearer.
            # For this test, we expect specific calls. If other calls happen, assertions on
            # call_args_list or part content will catch it.
            # Let's return a generic mock for unexpected calls to avoid crashing the test prematurely
            # and allow other assertions to pinpoint the issue.
            generic_mock_part = MagicMock(
                spec=gemini_types.Part, name=f"GenericTextPart_{text[:10]}"
            )
            generic_mock_part.text = text  # Store text for easier debugging if needed
            return generic_mock_part

        mock_part_from_text.side_effect = from_text_side_effect

        # Act: Call the method under test
        contents = await pipe_instance._genai_contents_from_messages(
            messages_body,
            messages_db,
            upload_documents,
            mock_event_emitter,
        )

        # Assert: Verify the outcome
        assert len(contents) == 1, "Should produce one content item"

        content_item = contents[0]
        assert content_item.role == "user", "Content item role should be 'user'"

        # --- Assertions for correct part segmentation and ordering (THESE WILL LIKELY FAIL with current code) ---
        # This is the core of the test, expecting proper interleaving.
        # Current code likely produces 2 parts: [YouTubePart, FullTextPart(including URL)]

        assert (
            len(content_item.parts) == 3
        ), f"Expected 3 parts (text, youtube, text), but got {len(content_item.parts)}. Parts: {[str(p) for p in content_item.parts]}"

        # Verify calls to types.Part.from_text
        # Expected: two calls, one for text_before_stripped, one for text_after_stripped.
        # Current code likely calls it once with the full user_content_string.
        expected_from_text_calls = [
            call(text=text_before_stripped),
            call(text=text_after_stripped),
        ]
        # Using assert_has_calls because the order of these text parts relative to each other
        # (if there were multiple text segments from markdown processing) might not be fixed,
        # but for this simple case, call_args_list would be stricter.
        # However, given the side_effect, checking the objects in content_item.parts is more direct.
        # Let's check call_count first, then the actual parts.
        assert (
            mock_part_from_text.call_count == 2
        ), f"Expected Part.from_text to be called 2 times, but was called {mock_part_from_text.call_count} times. Calls: {mock_part_from_text.call_args_list}"

        # Assert the content and order of parts
        # Part 1: Text before the YouTube link
        assert (
            content_item.parts[0] is mock_text_part_before_obj
        ), f"Part 0 was not the expected 'text_before' mock. Got: {content_item.parts[0]}"

        # Part 2: The YouTube link
        youtube_part = content_item.parts[1]
        assert isinstance(
            youtube_part, gemini_types.Part
        ), f"Part 1 is not a gemini_types.Part. Got: {type(youtube_part)}"
        assert hasattr(
            youtube_part, "file_data"
        ), "YouTube part (Part 1) should have 'file_data' attribute"
        assert (
            youtube_part.file_data is not None
        ), "YouTube part's file_data should not be None"
        assert (
            youtube_part.file_data.file_uri == youtube_url
        ), f"YouTube part URI mismatch. Expected: {youtube_url}, Got: {youtube_part.file_data.file_uri}"
        # Ensure it's not accidentally a text part
        assert (
            not hasattr(youtube_part, "text") or youtube_part.text is None
        ), "YouTube part should not have a 'text' attribute or it should be None"

        # Part 3: Text after the YouTube link
        assert (
            content_item.parts[2] is mock_text_part_after_obj
        ), f"Part 2 was not the expected 'text_after' mock. Got: {content_item.parts[2]}"

        # Ensure event_emitter was not called for this simple case
        mock_event_emitter.assert_not_called()


@pytest.mark.asyncio
async def test_genai_contents_from_messages_user_text_with_pdf(pipe_instance_fixture):
    """
    Tests conversion of a user message with text and an attached PDF file.
    """
    pipe_instance, MockedGenAIClientConstructor = pipe_instance_fixture

    # Arrange: Inputs
    user_text_content = "Please analyze this PDF."
    pdf_file_id = "test-pdf-id-001"
    pdf_file_name = "mydoc.pdf"
    fake_pdf_bytes = b"%PDF-1.4 fake content..."
    pdf_mime_type = "application/pdf"

    messages_body = [{"role": "user", "content": user_text_content}]

    # Construct messages_db with file attachment info
    messages_db = [
        {
            "id": "user-msg-db-id-1",
            "parentId": None,
            "childrenIds": [],
            "role": "user",
            "content": user_text_content,
            "timestamp": 1620000000,
            "files": [
                {
                    "id": "attachment-id-pdf-1",
                    "type": "file",
                    "file": {
                        "id": pdf_file_id,
                        "name": pdf_file_name,
                        "size": len(fake_pdf_bytes),
                        "timestamp": 1620000000,
                    },
                    "references": [],
                    "tool_code_id": None,
                }
            ],
        }
    ]
    upload_documents = True
    mock_event_emitter = AsyncMock()

    # Mock objects for parts
    mock_pdf_part_obj = MagicMock(spec=gemini_types.Part, name="PdfPart")
    mock_text_part_obj = MagicMock(spec=gemini_types.Part, name="TextPart")

    # Patch _get_file_data, Part.from_bytes, and Part.from_text
    with patch.object(
        Pipe, "_get_file_data", return_value=(fake_pdf_bytes, pdf_mime_type)
    ) as mock_get_file_data, patch(
        "plugins.pipes.gemini_manifold.types.Part.from_bytes",
        return_value=mock_pdf_part_obj,
    ) as mock_part_from_bytes, patch(
        "plugins.pipes.gemini_manifold.types.Part.from_text",
        return_value=mock_text_part_obj,
    ) as mock_part_from_text:

        # Act
        contents = await pipe_instance._genai_contents_from_messages(
            messages_body,
            messages_db,
            upload_documents,
            mock_event_emitter,
        )

        # Assert
        mock_get_file_data.assert_called_once_with(pdf_file_id)
        mock_part_from_bytes.assert_called_once_with(
            data=fake_pdf_bytes, mime_type=pdf_mime_type
        )
        mock_part_from_text.assert_called_once_with(text=user_text_content)

        assert len(contents) == 1
        user_content_obj = contents[0]
        assert user_content_obj.role == "user"
        assert len(user_content_obj.parts) == 2

        # Check order: PDF part should be first, then text part
        assert (
            user_content_obj.parts[0] is mock_pdf_part_obj
        ), "First part should be the PDF mock."
        assert (
            user_content_obj.parts[1] is mock_text_part_obj
        ), "Second part should be the text mock."

        mock_event_emitter.assert_not_called()


# endregion Test _genai_contents_from_messages


def teardown_module(module):
    """Cleans up sys.modules after tests in this file are done."""
    del sys.modules["open_webui.models.chats"]
    del sys.modules["open_webui.models.files"]
    del sys.modules["open_webui.models.functions"]
    del sys.modules["open_webui.storage.provider"]

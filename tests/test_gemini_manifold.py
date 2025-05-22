import pytest
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
from unittest.mock import ANY


# Helper async generator for mocking streaming responses
async def async_iter(items):
    for item in items:
        yield item


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
        "ENABLE_URL_CONTEXT_TOOL": False,  # Default to False for most existing tests
    }


# Helper to setup Pipe instance for tests that need it
# This reduces boilerplate in each test function.
@pytest.fixture
def pipe_instance_fixture(mock_pipe_valves_data):
    # Create a fresh copy of valves_data for each test to avoid modification pollution
    current_test_valves_data = mock_pipe_valves_data.copy()

    mock_gemini_client_instance = MagicMock()
    # Configure the client instance to have the 'aio.models' attribute structure
    mock_gemini_client_instance.aio = MagicMock()
    mock_gemini_client_instance.aio.models = MagicMock()
    mock_gemini_client_instance.aio.models.generate_content_stream = AsyncMock()
    mock_gemini_client_instance.aio.models.generate_content = AsyncMock()

    # Reset mocks that might be called during Pipe instantiation or setup
    mock_functions_module.Functions.get_function_valves_by_id.reset_mock()
    mock_functions_module.Functions.get_function_valves_by_id.return_value = (
        current_test_valves_data
    )

    with patch(
        "plugins.pipes.gemini_manifold.genai.Client",
        return_value=mock_gemini_client_instance,
    ) as MockedGenAIClientConstructor, patch.object(
        Pipe, "_add_log_handler", MagicMock()
    ) as mock_internal_add_log_handler, patch(
        "sys.stdout", MagicMock()  # Suppress print/log output during setup
    ):
        pipe = Pipe()
        # Yield a dictionary containing the pipe instance and the mock client
        # This allows tests to access the mock client directly if needed
        yield {
            "pipe": pipe,
            "mock_client": mock_gemini_client_instance,
            "mock_valves_data": current_test_valves_data,
            "MockedGenAIClientConstructor": MockedGenAIClientConstructor,
            "mock_internal_add_log_handler": mock_internal_add_log_handler,
        }


def test_pipe_initialization_with_api_key(mock_pipe_valves_data):
    # This test is about the initialization process itself.
    # We use the raw mock_pipe_valves_data here.
    valves_data_for_this_test = mock_pipe_valves_data.copy()

    mock_gemini_client_instance = MagicMock()
    mock_functions_module.Functions.get_function_valves_by_id.reset_mock()
    mock_functions_module.Functions.get_function_valves_by_id.return_value = (
        valves_data_for_this_test
    )

    with patch(
        "plugins.pipes.gemini_manifold.genai.Client",
        return_value=mock_gemini_client_instance,
    ) as MockedGenAIClientConstructor, patch.object(
        Pipe, "_add_log_handler", MagicMock()
    ) as mock_internal_add_log_handler, patch(
        "sys.stdout", MagicMock()
    ):
        # We need to re-initialize Pipe.valves for this specific test context
        # as pipe_instance_fixture might have already set it up with its own copy
        pipe_instance_valves = Pipe.Valves(**valves_data_for_this_test)
        
        # Temporarily patch self.valves for the _get_or_create_genai_client call
        # within the Pipe constructor if it's called there.
        # More robustly, ensure the fixture provides a fresh Pipe instance for each test.
        # The fixture approach is better. This test can use the fixture and assert on its state.

        # Re-instantiate Pipe to test its __init__ logic directly
        # Note: The fixture `pipe_instance_fixture` also creates a Pipe instance.
        # If this test is solely about __init__, it should manage its own Pipe creation
        # or rely on the fixture and assert post-conditions.
        # Let's use a fresh instance for clarity on __init__ testing.
        
        pipe_instance = Pipe() # This will use the mocked get_function_valves_by_id

        # Assertions
        mock_functions_module.Functions.get_function_valves_by_id.assert_called_once_with(
            "gemini_manifold_google_genai"
        )
        assert isinstance(pipe_instance.valves, Pipe.Valves)
        assert pipe_instance.valves.GEMINI_API_KEY == valves_data_for_this_test["GEMINI_API_KEY"]
        
        # _add_log_handler is called in pipes() and pipe() methods, not __init__ usually.
        # If it's called by _get_or_create_genai_client or similar during __init__, this is fine.
        # Based on current code, _get_or_create_genai_client is NOT called in __init__.
        # It's called in _get_user_client, which is called in pipe(), or in _get_genai_models (in pipes()).
        # So, mock_internal_add_log_handler.assert_called_once() might be too strict if __init__ doesn't log.
        # Let's assume the fixture handles the default client creation path which might log.
        # The fixture already asserts these, so this test can focus on values.

        # The client is created on first access via _get_user_client or _get_genai_models.
        # The fixture creates a default client. Let's verify the arguments if it was created.
        # If the fixture is used, it handles this. If Pipe() is called directly, client isn't made yet.
        # For this specific test, let's assume we want to check the first client creation.
        # We can trigger it by calling a method that uses the client.
        # However, the prompt is about Pipe initialization.
        # The fixture `pipe_instance_fixture` already tests the client creation part.
        # This test should focus on the state of `pipe_instance.valves`.
        # The MockedGenAIClientConstructor assertion belongs more to the fixture setup verification.
        
        # If the goal is to test the client creation path triggered by __init__ (if any),
        # then the mock setup needs to align with that.
        # Current Pipe.__init__ just sets self.valves.
        # The client is lazy-loaded or pre-warmed by the fixture.
        # Let's ensure this test is meaningful for __init__ specifically.
        
        # If the client *is* created by __init__ (e.g., self.clients["default"] = ... in __init__)
        # then MockedGenAIClientConstructor.assert_called_once_with(...) is correct.
        # Based on the provided code, `self.clients` is not a thing, client is created on demand.
        # The fixture warms up a client by calling _get_or_create_genai_client
        # Let's rely on the fixture for client creation tests and keep this focused.

        # This test can be simplified if pipe_instance_fixture is used and assertions are made on its state.
        # Let's use the fixture provided state to assert.
        
        setup = pipe_instance_fixture # This re-runs the fixture for this test
        
        setup["MockedGenAIClientConstructor"].assert_called_once_with(
            api_key=valves_data_for_this_test["GEMINI_API_KEY"],
            http_options=gemini_types.HttpOptions(
                base_url=valves_data_for_this_test["GEMINI_API_BASE_URL"]
            ),
            vertexai=ANY, # Added due to changes in _get_or_create_genai_client
            project=ANY,
            location=ANY,
        )
        assert setup["pipe"].valves.GEMINI_API_KEY == valves_data_for_this_test["GEMINI_API_KEY"]
        setup["mock_internal_add_log_handler"].assert_called_once() # From fixture setup


@pytest.mark.asyncio
async def test_genai_contents_from_messages_simple_user_text(pipe_instance_fixture):
    pipe_details = pipe_instance_fixture
    pipe_instance = pipe_details["pipe"]

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
    pipe_details = pipe_instance_fixture
    pipe_instance = pipe_details["pipe"]

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

        # --- Assertions for correct part segmentation and ordering ---
        assert (
            len(content_item.parts) == 3
        ), f"Expected 3 parts (text, youtube, text), but got {len(content_item.parts)}. Parts: {[str(p) for p in content_item.parts]}"

        # Verify calls to types.Part.from_text
        assert (
            mock_part_from_text.call_count == 2
        ), f"Expected Part.from_text to be called 2 times, but was called {mock_part_from_text.call_count} times. Calls: {mock_part_from_text.call_args_list}"

        # Assert the content and order of parts
        assert (
            content_item.parts[0] is mock_text_part_before_obj
        ), f"Part 0 was not the expected 'text_before' mock. Got: {content_item.parts[0]}"

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
        assert (
            not hasattr(youtube_part, "text") or youtube_part.text is None
        ), "YouTube part should not have a 'text' attribute or it should be None"

        assert (
            content_item.parts[2] is mock_text_part_after_obj
        ), f"Part 2 was not the expected 'text_after' mock. Got: {content_item.parts[2]}"

        mock_event_emitter.assert_not_called()


@pytest.mark.asyncio
async def test_genai_contents_from_messages_user_text_with_pdf(pipe_instance_fixture):
    pipe_details = pipe_instance_fixture
    pipe_instance = pipe_details["pipe"]

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
            "content": user_text_content,  # Should match messages_body content
            "timestamp": 1620000000,
            "files": [
                {
                    "id": "attachment-id-pdf-1",  # ID of the attachment link
                    "type": "file",
                    "file": {  # FileInfoTD
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


# --- Tests for URL Context Tool ---
@pytest.mark.asyncio
async def test_url_context_tool_enabled_compatible_model(pipe_instance_fixture):
    pipe_details = pipe_instance_fixture
    pipe_instance = pipe_details["pipe"]
    mock_client = pipe_details["mock_client"]
    pipe_details["mock_valves_data"]["ENABLE_URL_CONTEXT_TOOL"] = True

    body = {
        "model": "models/gemini-2.5-flash-preview-05-20", # Compatible model
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    user_data = {"id": "test_user_id", "email": "test@example.com", "valves": None}
    request_mock = MagicMock()
    event_emitter_mock = AsyncMock()
    metadata = {"chat_id": "chat123", "message_id": "msg123", "features": {}}

    # Mock the streaming response
    mock_response_chunk = MagicMock(spec=gemini_types.GenerateContentResponse)
    mock_candidate = MagicMock(spec=gemini_types.Candidate)
    mock_candidate.content = gemini_types.Content(parts=[gemini_types.Part(text="response")])
    mock_candidate.finish_reason = gemini_types.FinishReason.STOP
    mock_candidate.url_context_metadata = None
    mock_candidate.grounding_metadata = None # Ensure this is None or a mock
    mock_response_chunk.candidates = [mock_candidate]
    mock_response_chunk.usage_metadata = MagicMock()
    mock_client.aio.models.generate_content_stream.return_value = async_iter([mock_response_chunk])

    # Call the pipe method
    async for _ in await pipe_instance.pipe(body, user_data, request_mock, event_emitter_mock, metadata):
        pass

    # Assert generate_content_stream was called
    mock_client.aio.models.generate_content_stream.assert_called_once()
    args, kwargs = mock_client.aio.models.generate_content_stream.call_args
    
    # Check that UrlContext tool is in the tools list
    generate_config = kwargs["config"]
    assert gemini_types.Tool(url_context=gemini_types.UrlContext()) in generate_config.tools
    
    # Check that no YouTube FileData part was created
    contents_passed = kwargs["contents"]
    for content_item in contents_passed:
        for part in content_item.parts:
            assert not (hasattr(part, "file_data") and part.file_data and "youtube.com" in part.file_data.file_uri), \
                   "YouTube FileData part should not be present without a YouTube URL"


@pytest.mark.asyncio
async def test_url_context_tool_enabled_incompatible_model(pipe_instance_fixture):
    pipe_details = pipe_instance_fixture
    pipe_instance = pipe_details["pipe"]
    mock_client = pipe_details["mock_client"]
    pipe_details["mock_valves_data"]["ENABLE_URL_CONTEXT_TOOL"] = True

    body = {
        "model": "models/gemini-1.0-pro",  # Incompatible model
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    # ... (rest of the setup is similar to the compatible model test)
    user_data = {"id": "test_user_id", "email": "test@example.com", "valves": None}
    request_mock = MagicMock()
    event_emitter_mock = AsyncMock()
    metadata = {"chat_id": "chat123", "message_id": "msg123", "features": {}}

    mock_response_chunk = MagicMock(spec=gemini_types.GenerateContentResponse) # Basic mock
    mock_candidate = MagicMock(spec=gemini_types.Candidate)
    mock_candidate.content = gemini_types.Content(parts=[gemini_types.Part(text="response")])
    mock_candidate.finish_reason = gemini_types.FinishReason.STOP
    mock_candidate.url_context_metadata = None
    mock_candidate.grounding_metadata = None
    mock_response_chunk.candidates = [mock_candidate]
    mock_response_chunk.usage_metadata = MagicMock()
    mock_client.aio.models.generate_content_stream.return_value = async_iter([mock_response_chunk])

    async for _ in await pipe_instance.pipe(body, user_data, request_mock, event_emitter_mock, metadata):
        pass
        
    mock_client.aio.models.generate_content_stream.assert_called_once()
    args, kwargs = mock_client.aio.models.generate_content_stream.call_args
    generate_config = kwargs["config"]
    assert gemini_types.Tool(url_context=gemini_types.UrlContext()) not in generate_config.tools


@pytest.mark.asyncio
async def test_url_context_tool_disabled(pipe_instance_fixture):
    pipe_details = pipe_instance_fixture
    pipe_instance = pipe_details["pipe"]
    mock_client = pipe_details["mock_client"]
    pipe_details["mock_valves_data"]["ENABLE_URL_CONTEXT_TOOL"] = False # Explicitly disabled

    body = {
        "model": "models/gemini-2.5-flash-preview-05-20", # Compatible model
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    # ... (rest of the setup)
    user_data = {"id": "test_user_id", "email": "test@example.com", "valves": None}
    request_mock = MagicMock()
    event_emitter_mock = AsyncMock()
    metadata = {"chat_id": "chat123", "message_id": "msg123", "features": {}}
    
    mock_response_chunk = MagicMock(spec=gemini_types.GenerateContentResponse) # Basic mock
    mock_candidate = MagicMock(spec=gemini_types.Candidate)
    mock_candidate.content = gemini_types.Content(parts=[gemini_types.Part(text="response")])
    mock_candidate.finish_reason = gemini_types.FinishReason.STOP
    mock_candidate.url_context_metadata = None
    mock_candidate.grounding_metadata = None
    mock_response_chunk.candidates = [mock_candidate]
    mock_response_chunk.usage_metadata = MagicMock()
    mock_client.aio.models.generate_content_stream.return_value = async_iter([mock_response_chunk])

    async for _ in await pipe_instance.pipe(body, user_data, request_mock, event_emitter_mock, metadata):
        pass

    mock_client.aio.models.generate_content_stream.assert_called_once()
    args, kwargs = mock_client.aio.models.generate_content_stream.call_args
    generate_config = kwargs["config"]
    assert gemini_types.Tool(url_context=gemini_types.UrlContext()) not in generate_config.tools

@pytest.mark.asyncio
async def test_youtube_url_handling_precedence_with_url_context(pipe_instance_fixture):
    pipe_details = pipe_instance_fixture
    pipe_instance = pipe_details["pipe"]
    mock_client = pipe_details["mock_client"]
    pipe_details["mock_valves_data"]["ENABLE_URL_CONTEXT_TOOL"] = True

    youtube_url = "https://www.youtube.com/watch?v=video123"
    other_url = "https://example.com/article"
    message_content = f"Check this YouTube video {youtube_url} and this article {other_url}"

    body = {
        "model": "models/gemini-2.5-flash-preview-05-20", # Compatible
        "messages": [{"role": "user", "content": message_content}],
        "stream": True,
    }
    # ... (rest of the setup)
    user_data = {"id": "test_user_id", "email": "test@example.com", "valves": None}
    request_mock = MagicMock()
    event_emitter_mock = AsyncMock()
    metadata = {"chat_id": "chat123", "message_id": "msg123", "features": {}}

    mock_response_chunk = MagicMock(spec=gemini_types.GenerateContentResponse) # Basic mock
    mock_candidate = MagicMock(spec=gemini_types.Candidate)
    mock_candidate.content = gemini_types.Content(parts=[gemini_types.Part(text="response")])
    mock_candidate.finish_reason = gemini_types.FinishReason.STOP
    mock_candidate.url_context_metadata = None
    mock_candidate.grounding_metadata = None
    mock_response_chunk.candidates = [mock_candidate]
    mock_response_chunk.usage_metadata = MagicMock()
    mock_client.aio.models.generate_content_stream.return_value = async_iter([mock_response_chunk])
    
    async for _ in await pipe_instance.pipe(body, user_data, request_mock, event_emitter_mock, metadata):
        pass

    mock_client.aio.models.generate_content_stream.assert_called_once()
    args, kwargs = mock_client.aio.models.generate_content_stream.call_args
    
    # Assert URL Context tool is present
    generate_config = kwargs["config"]
    assert gemini_types.Tool(url_context=gemini_types.UrlContext()) in generate_config.tools
    
    # Assert YouTube URL is a FileData part
    contents_passed = kwargs["contents"]
    found_youtube_file_data = False
    found_other_url_in_text = False

    for content_item in contents_passed:
        if content_item.role == "user":
            for part in content_item.parts:
                if hasattr(part, "file_data") and part.file_data and part.file_data.file_uri == youtube_url:
                    found_youtube_file_data = True
                if hasattr(part, "text") and part.text and other_url in part.text:
                    found_other_url_in_text = True
    
    assert found_youtube_file_data, "YouTube URL should be processed as FileData"
    assert found_other_url_in_text, "Non-YouTube URL should be part of a text part"


@pytest.mark.asyncio
async def test_url_context_metadata_processing(pipe_instance_fixture):
    pipe_details = pipe_instance_fixture
    pipe_instance = pipe_details["pipe"]
    mock_client = pipe_details["mock_client"]
    pipe_details["mock_valves_data"]["ENABLE_URL_CONTEXT_TOOL"] = True

    body = {
        "model": "models/gemini-2.5-pro-preview-05-06", # Compatible
        "messages": [{"role": "user", "content": "Summarize https://example.com/page1"}],
        "stream": True,
    }
    user_data = {"id": "test_user_id", "email": "test@example.com", "valves": None}
    request_mock = MagicMock()
    event_emitter_mock = AsyncMock()
    metadata = {"chat_id": "chat123", "message_id": "msg123", "features": {}}

    # --- Mock response with URL context metadata ---
    mock_retrieved_url1 = MagicMock(spec=gemini_types.UrlContextRetrievedUrl)
    mock_retrieved_url1.url = "https://example.com/retrieved1"
    mock_retrieved_url1.title = "Retrieved Page 1"

    mock_url_metadata = MagicMock(spec=gemini_types.UrlContextMetadata)
    mock_url_metadata.retrieved_urls = [mock_retrieved_url1]

    mock_candidate = MagicMock(spec=gemini_types.Candidate)
    mock_candidate.url_context_metadata = mock_url_metadata
    mock_candidate.finish_reason = gemini_types.FinishReason.STOP
    # Ensure content and parts exist, even if minimal
    mock_part = MagicMock(spec=gemini_types.Part)
    mock_part.text = "Model response based on URL."
    mock_content = MagicMock(spec=gemini_types.Content)
    mock_content.parts = [mock_part]
    mock_candidate.content = mock_content
    mock_candidate.grounding_metadata = None # Important for _do_post_processing logic

    mock_final_response_chunk = MagicMock(spec=gemini_types.GenerateContentResponse)
    mock_final_response_chunk.candidates = [mock_candidate]
    mock_final_response_chunk.usage_metadata = MagicMock() # For _get_usage_data_event
    mock_final_response_chunk.usage_metadata.prompt_token_count = 10
    mock_final_response_chunk.usage_metadata.candidates_token_count = 20
    mock_final_response_chunk.usage_metadata.total_token_count = 30
    
    # Simulate a stream with one chunk that contains all data
    mock_client.aio.models.generate_content_stream.return_value = async_iter([mock_final_response_chunk])

    # Call the pipe method and consume the stream
    async for _ in await pipe_instance.pipe(body, user_data, request_mock, event_emitter_mock, metadata):
        pass

    # Assert event_emitter was called with chat:url_context
    # event_emitter_mock.assert_any_call(...) doesn't quite work for complex dicts easily.
    # We need to find the call in call_args_list.
    
    emitted_url_context_event = None
    for call_item in event_emitter_mock.call_args_list:
        args, _ = call_item
        event = args[0] # The event is the first positional argument
        if event.get("type") == "chat:url_context":
            emitted_url_context_event = event
            break
    
    assert emitted_url_context_event is not None, "chat:url_context event was not emitted"
    assert emitted_url_context_event["data"]["retrieved_urls"] == [
        {"url": "https://example.com/retrieved1", "title": "Retrieved Page 1"}
    ]

    # Also ensure usage data event was called, as it's part of _do_post_processing
    emitted_usage_event = None
    for call_item in event_emitter_mock.call_args_list:
        args, _ = call_item
        event = args[0]
        if event.get("type") == "chat:completion" and "usage" in event.get("data", {}):
            emitted_usage_event = event
            break
    assert emitted_usage_event is not None, "chat:completion event with usage data was not emitted"

# --- End of Tests for URL Context Tool ---


def teardown_module(module):
    """Cleans up sys.modules after tests in this file are done."""
    del sys.modules["open_webui.models.chats"]
    del sys.modules["open_webui.models.files"]
    del sys.modules["open_webui.models.functions"]
    del sys.modules["open_webui.storage.provider"]

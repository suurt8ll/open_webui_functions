import argparse
import asyncio
import hashlib
import json
import os
import signal
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import aiofiles
import aiohttp
from dotenv import load_dotenv
from loguru import logger
from watchfiles import Change, awatch

# --- Configuration & State Management ---


class ApiState(Enum):
    """Represents the perceived state of the remote API."""

    UNKNOWN = auto()
    AVAILABLE = auto()
    UNAVAILABLE = auto()


@dataclass(frozen=True)
class Config:
    """Typed configuration loaded from environment variables."""

    api_endpoint: str  # Calculated from HOST and PORT
    api_key: str
    filepaths: list[str]
    polling_interval: int
    one_time_run: bool


def instance_state_file(config: Config) -> str:
    """Path of the sync-state file for a given config (endpoint+key scoped).

    The suffix prevents false-positive sync states when switching between
    different OWUI environments (e.g., local vs remote).
    """
    instance_id = hashlib.sha256(f'{config.api_endpoint}{config.api_key}'.encode()).hexdigest()[:8]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    state_dir = os.path.join(current_dir, '.dev_state')
    return os.path.join(state_dir, f'hashes_{instance_id}.json')


def load_config(env_path: str | None = None) -> Config:
    """Loads and validates required environment variables into a Config object."""
    if env_path:
        abs_env_path = os.path.abspath(env_path)
        if not os.path.isfile(abs_env_path):
            raise ValueError(f'Custom .env file not found at: {abs_env_path}')
        # An explicitly provided env file is the authoritative source: it wins
        # over any variables already present in the environment. Without this,
        # pre-exported vars would silently shadow the file and produce confusing
        # behavior when switching between env files in one shell.
        load_dotenv(dotenv_path=abs_env_path, override=True)
        base_dir = os.path.dirname(abs_env_path)
    else:
        load_dotenv()
        base_dir = os.getcwd()

    required_vars = ['API_KEY', 'FILEPATHS']
    env_vars: dict[str, Any] = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            logger.critical(f'Missing required environment variable: {var}')
            raise ValueError(f'Missing environment variable: {var}')
        env_vars[var] = value

    protocol = os.getenv('PROTOCOL', 'http').lower()
    host = os.getenv('HOST', '127.0.0.1')
    port = os.getenv('PORT', '8080')
    api_endpoint = f'{protocol}://{host}:{port}'

    raw_paths = [path.strip() for path in env_vars['FILEPATHS'].split(',')]
    abs_paths = [os.path.abspath(os.path.join(base_dir, p)) for p in raw_paths if p]

    one_time_run = os.getenv('ONE_TIME_RUN', 'false').lower() in (
        'true',
        '1',
        'yes',
        'y',
    )

    return Config(
        api_endpoint=api_endpoint,
        api_key=env_vars['API_KEY'],
        filepaths=abs_paths,
        polling_interval=int(os.getenv('POLLING_INTERVAL', '60')),
        one_time_run=one_time_run,
    )


# --- File Utilities ---


async def file_hash(filename: str) -> str | None:
    """
    Calculate the SHA256 hash of a file asynchronously.
    Returns None if the file cannot be read.
    """
    hasher = hashlib.sha256()
    try:
        async with aiofiles.open(filename, 'rb') as file:
            while chunk := await file.read(4096):
                hasher.update(chunk)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.exception(f'Error reading file {filename}: {e}')
        return None
    return hasher.hexdigest()


async def extract_metadata_from_file(filepath: str) -> dict[str, str]:
    """
    Extracts metadata from the first docstring in a Python file.
    """
    metadata = {}
    try:
        async with aiofiles.open(filepath, encoding='utf-8') as file:
            content = await file.read()

        if not content.strip().startswith('"""'):
            return {}

        doc_start = content.find('"""')
        doc_end = content.find('"""', doc_start + 3)

        if doc_end == -1:
            return {}

        docstring = content[doc_start + 3 : doc_end]
        for line in docstring.splitlines():
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()

    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f'Metadata extraction error for {filepath}: {e}')

    return metadata


# --- API Client ---


class OwuiApiClient:
    """A client to interact with the OWUI functions API."""

    def __init__(self, session: aiohttp.ClientSession, root_url: str, api_key: str):
        self._session = session
        self._root_url = root_url
        self._base_api_url = f'{root_url}/api/v1'
        self._api_key = api_key
        self._headers = {'Authorization': f'Bearer {self._api_key}'}

    async def check_availability(self) -> bool:
        """Checks if the API is healthy and responsive."""
        try:
            async with self._session.get(
                f'{self._root_url}/health', timeout=aiohttp.ClientTimeout(total=3)
            ) as response:
                return response.status == 200
        except (TimeoutError, aiohttp.ClientError):
            return False
        except Exception as e:
            logger.exception(f'Unexpected error checking API: {e}')
            return False

    async def function_exists(self, function_id: str) -> bool:
        """Checks if a function with the given ID exists."""
        try:
            async with self._session.get(f'{self._base_api_url}/functions/', headers=self._headers) as response:
                response.raise_for_status()
                functions = await response.json()
                return any(f['id'] == function_id for f in functions)
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.error(f'Error checking function existence: {e}')
            return False
        except Exception as e:
            logger.exception(f'Unexpected error checking function: {e}')
            return False

    async def _create_function(
        self,
        filepath: str,
        function_id: str,
        function_name: str,
        description: str,
        code_content: str,
    ) -> None:
        """Helper to create a new remote function."""
        payload = {
            'id': function_id,
            'name': function_name,
            'content': code_content,
            'meta': {'description': description, 'manifest': {}},
        }
        async with self._session.post(
            f'{self._base_api_url}/functions/create',
            headers=self._headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            logger.info(f"✅ Created new function '{function_id}' from {filepath}")

    async def update_or_create_function(
        self,
        filepath: str,
        function_id: str,
        function_name: str,
        description: str,
    ) -> bool:
        """Updates a remote function, or creates it if it doesn't exist. Returns True on success."""
        try:
            async with aiofiles.open(filepath, encoding='utf-8') as file:
                code_content = await file.read()
        except OSError as e:
            logger.error(f'Failed to read file {filepath} before update: {e}')
            return False

        try:
            payload = {
                'id': function_id,
                'name': function_name,
                'content': code_content,
                'meta': {'description': description, 'manifest': {}},
            }

            async with self._session.post(
                f'{self._base_api_url}/functions/id/{function_id}/update',
                headers=self._headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                logger.info(f"🔄 Updated function '{function_id}' from {filepath}")
                return True

        except aiohttp.ClientResponseError as e:
            if e.status in {400, 404}:
                logger.warning(
                    f"Function '{function_id}' not found or failed to update (Status: {e.status})."
                    ' Attempting to create it.'
                )
                try:
                    await self._create_function(filepath, function_id, function_name, description, code_content)
                    return True
                except (TimeoutError, aiohttp.ClientError) as create_err:
                    logger.error(f'Creation failed for {filepath}: {create_err}')
                except Exception:
                    logger.exception(f'Unexpected error creating function {function_id}')
            else:
                logger.error(f'Update failed for {filepath}: {e}')
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.error(f'Update failed for {filepath}: {type(e).__name__}: {e}. Check network/API status.')
        except Exception as e:
            logger.exception(f'Unexpected error updating function: {e}')

        return False


# --- Main Application ---


class FunctionUpdater:
    """The main application class that orchestrates file monitoring and updates."""

    def __init__(self, config: Config, shutdown_event: asyncio.Event):
        self.config = config
        self.shutdown_event = shutdown_event
        self.api_state = ApiState.UNKNOWN
        self.missing_file_logged: set[str] = set()
        self._process_lock = asyncio.Lock()

        self._state_file = instance_state_file(config)

        self.file_hashes: dict[str, str | None] = self._load_state()

        # Ensure all currently tracked files are in the hashes dictionary
        for path in config.filepaths:
            if path not in self.file_hashes:
                self.file_hashes[path] = None

    def _load_state(self) -> dict[str, str | None]:
        """Loads the previous file hashes from disk on startup."""
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, encoding='utf-8') as f:
                    state = json.load(f)
                    logger.info(f'💾 Loaded previous state from {self._state_file}')
                    return state
            except json.JSONDecodeError:
                logger.warning('Corrupted state file found. Starting fresh.')
            except Exception as e:
                logger.warning(f'Failed to load state: {e}. Starting fresh.')
        return {}

    async def _save_state(self) -> None:
        """Saves the current file hashes to disk asynchronously."""
        try:
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            async with aiofiles.open(self._state_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.file_hashes, indent=2))
        except Exception as e:
            logger.error(f'Failed to save state to disk: {e}')

    async def _check_api_status(self, client: OwuiApiClient) -> None:
        """Checks API availability and updates the state, logging changes."""
        is_available = await client.check_availability()
        if is_available:
            if self.api_state != ApiState.AVAILABLE:
                logger.info('API is available. Processing files...')
                self.api_state = ApiState.AVAILABLE
        else:
            if self.api_state != ApiState.UNAVAILABLE:
                logger.warning('API is unavailable. Will keep retrying...')
                self.api_state = ApiState.UNAVAILABLE

    async def _process_file(self, path: str, client: OwuiApiClient) -> None:
        """Processes a single file, checking for changes and updating if necessary."""
        async with self._process_lock:
            if not os.path.exists(path):
                if path not in self.missing_file_logged:
                    logger.warning(f'File not found: {path}')
                    self.missing_file_logged.add(path)
                return

            if path in self.missing_file_logged:
                self.missing_file_logged.remove(path)

            current_hash = await file_hash(path)
            if not current_hash:
                return

            if current_hash == self.file_hashes.get(path):
                return  # No change, skip processing

            logger.debug(f'File change detected for {path}')
            metadata = await extract_metadata_from_file(path)
            function_id = metadata.get('id')
            function_name = metadata.get('title')
            description = metadata.get('description')

            if not all([function_id, function_name, description]):
                logger.warning(f"Skipping {path}: Missing required metadata ('id', 'title', 'description').")
                return

            assert function_id is not None
            assert function_name is not None
            assert description is not None

            # Attempt update. Only save the hash if the update succeeded.
            success = await client.update_or_create_function(path, function_id, function_name, description)

            if success:
                self.file_hashes[path] = current_hash
                # Save state to disk immediately upon success. Because we are inside
                # self._process_lock, this file write operation is race-condition safe.
                await self._save_state()

    async def _watch_loop(self, client: OwuiApiClient) -> None:
        """Instantly reacts to file changes using OS-level events."""
        # Bypassing the file watcher entirely saves OS-level resources and thread allocation
        # since we won't be lingering around waiting for updates.
        if self.config.one_time_run:
            return

        # awatch crashes the whole TaskGroup (killing the poll loop too) if a
        # watched directory doesn't exist. Skip missing dirs with a warning;
        # the poll loop still picks up files there once they appear.
        directories_to_watch: set[str] = set()
        for path in self.config.filepaths:
            directory = os.path.dirname(path) or '.'
            if os.path.isdir(directory):
                directories_to_watch.add(directory)
            else:
                logger.warning(
                    f'Watch directory does not exist, not watching it (poll fallback still applies): {directory}'
                )

        if not directories_to_watch:
            logger.warning('No watchable directories; relying on the poll loop only.')
            return

        logger.info(f'👀 Started file watcher on {len(self.config.filepaths)} files...')

        async for changes in awatch(*directories_to_watch, stop_event=self.shutdown_event):
            for change_type, path in changes:
                if path in self.config.filepaths and change_type != Change.deleted:
                    await self._process_file(path, client)

    async def _poll_loop(self, client: OwuiApiClient) -> None:
        """A slow polling loop that acts as a fallback and sync mechanism."""
        while not self.shutdown_event.is_set():
            await self._check_api_status(client)

            if self.api_state == ApiState.AVAILABLE:
                for path in self.config.filepaths:
                    await self._process_file(path, client)

                if self.config.one_time_run:
                    logger.info('One-time sync complete. Shutting down...')
                    self.shutdown_event.set()
                    break

            # When the API is offline or its state is unknown, we poll frequently (1s)
            # to reconnect as soon as it comes back up. Once available, we respect
            # the user-configured interval to avoid unnecessary overhead.
            current_interval = self.config.polling_interval if self.api_state == ApiState.AVAILABLE else 1

            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=current_interval)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def run(self) -> None:
        """The main execution loop."""
        async with aiohttp.ClientSession() as session:
            api_client = OwuiApiClient(session, self.config.api_endpoint, self.config.api_key)

            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._poll_loop(api_client))
                    tg.create_task(self._watch_loop(api_client))
            except Exception as e:
                if not isinstance(e, asyncio.CancelledError):
                    logger.exception(f'An error occurred during execution: {e}')


# --- Status Mode ---


async def print_sync_status(config: Config) -> int:
    """Print per-file sync state (local file hash vs last-successfully-synced
    hash) without touching the API or starting any watchers. Useful for agents
    and humans to verify what is in sync at a glance."""
    state_file = instance_state_file(config)
    try:
        async with aiofiles.open(state_file, encoding='utf-8') as f:
            stored_hashes: dict[str, str | None] = json.loads(await f.read())
    except FileNotFoundError:
        stored_hashes = {}
        logger.warning(f'No sync state file yet ({state_file}) — nothing has been synced for this endpoint.')
    except Exception as e:
        logger.warning(f'Could not read sync state ({e}); treating as empty.')
        stored_hashes = {}

    rows: list[tuple[str, str, str]] = []
    for path in config.filepaths:
        display = os.path.relpath(path)
        if not os.path.isfile(path):
            rows.append((display, '-', 'MISSING'))
            continue
        metadata = await extract_metadata_from_file(path)
        function_id = metadata.get('id') or '?'
        current_hash = await file_hash(path)
        if current_hash is None:
            status = 'MISSING'
        elif path not in stored_hashes:
            status = 'NEVER SYNCED'
        elif stored_hashes[path] == current_hash:
            status = 'IN SYNC'
        else:
            status = 'MODIFIED (sync pending)'
        rows.append((display, function_id, status))

    width_path = max(len(r[0]) for r in rows) if rows else 0
    width_id = max(len(r[1]) for r in rows) if rows else 0
    print(f'Target: {config.api_endpoint}')
    print(f'{"FILE".ljust(width_path)}  {"FUNCTION ID".ljust(width_id)}  STATUS')
    for display, function_id, status in rows:
        print(f'{display.ljust(width_path)}  {function_id.ljust(width_id)}  {status}')
    return 0


async def main_wrapper() -> None:
    """Sets up signal handling and runs the main application."""
    parser = argparse.ArgumentParser(description='OWUI Function Updater Utility')
    parser.add_argument('--env', type=str, help='Path to a custom .env file (optional)')
    parser.add_argument(
        '--status',
        action='store_true',
        help='Print the per-file sync state for the given endpoint and exit (no watching, no API writes).',
    )
    args = parser.parse_args()

    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        logger.warning(f'Received signal {signum}. Initiating shutdown...')
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler, sig, None)

    try:
        # Pass the parsed env path to the config loader
        config = load_config(args.env)
        logger.info(
            f'🎯 Target: {config.api_endpoint} | {len(config.filepaths)} file(s) | '
            f'config source: {os.path.abspath(args.env) if args.env else "ambient environment"}'
        )
        if args.status:
            await print_sync_status(config)
            return
        updater = FunctionUpdater(config, shutdown_event)
        await updater.run()
    except ValueError as e:
        logger.critical(e)
    except asyncio.CancelledError:
        logger.info('Main task cancelled.')
    except Exception:
        logger.exception('An unexpected error occurred in the main application:')
    finally:
        logger.info('Shutdown complete.')


if __name__ == '__main__':
    logger.info('Starting function updater utility...')
    asyncio.run(main_wrapper())

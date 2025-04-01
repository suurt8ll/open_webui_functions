# Open WebUI Plugin System

## Introduction

Open WebUI features a plugin system that allows for customization and extension of its functionality. The plugin system is primarily implemented in the backend using Python and is based on dynamically loading and executing code. Plugins allow you to modify the behavior of Open WebUI without directly modifying the core application code.

A crucial convention: **Plugin classes *must* be named according to their type: `Filter`, `Pipe`, or `Action`. The backend relies on these exact names for plugin loading and execution.** Any other class name will prevent the plugin from functioning correctly.

## Plugin Types

The plugin system supports three identified plugin types, referred to as "functions" internally:

*   **Filter Plugins:** These plugins modify request and response data at various stages of the LLM interaction. See the [Filter Plugin Reference](plugins/filter.md) for detailed information.
*   **Pipe Plugins:** These plugins allow you to integrate custom models and behaviors into Open WebUI. See the [Pipe Plugin Reference](plugins/pipe.md) for detailed information.
*   **Action Plugins:** *The functionality and methods associated with Action plugins are not yet fully documented.*

## Plugin Structure

While the specific structure varies by plugin type, all plugins generally follow this pattern:

*   **Code File:** A Python file containing the plugin's logic.
*   **Class Definition:** A class (e.g., `Filter`, `Pipe`, `Action`) that encapsulates the plugin's functionality. **Ensure the class name exactly matches the plugin type (e.g., `Filter` for a filter plugin).**
*   **Methods:** Methods within the class that are called at specific points in the Open WebUI workflow.
*   **Frontmatter:** Metadata at the top of the file (e.g., dependencies, priority).

## Plugin Lifecycle

The plugin lifecycle describes the different stages a plugin goes through from installation to uninstallation. Understanding this lifecycle is crucial for developing robust and well-behaved plugins.

*   **Installation:** Plugins can be installed either through the Open WebUI API or via the administrative front-end. In the UI, administrators can navigate to the `Functions` subpage of the Admin Panel and click the "+" icon to create a new plugin. This involves providing a title, ID, description, and the actual code for the plugin. After filling in the information the user can click the "Save" button.

*   **Activation/Enabling:** By default, a newly installed plugin is disabled. To activate a plugin, it must be explicitly enabled. This can be done either through the UI by toggling the "enable" button next to the plugin on the `Functions` subpage, or programmatically through the API by setting a specific key-value pair to `True` (the exact key name needs to be determined). Only enabled plugins have an effect on the functioning of Open WebUI. Disabled plugins are initialized but their code is not executed during normal operation.

*   **Configuration:** Plugins that define valves (configuration options) can have their settings adjusted after installation. In the UI, a gear icon appears next to the function. Clicking this icon opens a window displaying the variables defined in the `Valves` Pydantic model, allowing administrators to input new values.

*   **Deactivation/Disabling:** Disabling a plugin through the toggle switch does *not* completely unload it from memory. While the plugin's code is no longer actively executed, variables stored within the plugin's `self` context persist even after disabling. The primary effect of disabling is that the plugin's functionality is removed from the system. For example, a disabled Pipe plugin will no longer appear in the model list, and a disabled Filter plugin will no longer modify requests or responses.

*   **Code Updates:** It's important to note that updating a plugin's code will reset any variables stored within the `self` context of the plugin. When a user edits the code for a plugin and saves the changes, the plugin is effectively re-initialized with the new code, clearing any previously stored state.

*   **Uninstallation/Deletion:** To completely remove a plugin, it must be uninstalled or deleted. This can be done either through the API or in the "Functions" subpage of the UI by clicking the "..." icon next to the function and selecting "Delete." Uninstalling a plugin completely destroys and unloads the plugin from the system, removing all associated code and data.

## Dependency Management

The plugin system supports specifying Python package dependencies in the plugin's frontmatter. The `install_frontmatter_requirements` function is used to install these dependencies. You can specify dependencies using standard Python package names (e.g., `requests`, `beautifulsoup4`).

## Priority

Plugins can be assigned a priority, which determines the order in which they are applied. Lower numbers indicate higher priority (i.e., they are executed earlier). This is important when multiple plugins need to modify the same data.

## Context Injection

The plugin methods receive a context dictionary containing information about the user (`__user__`), the request (`__request__`), the model (`__model__`), and other relevant data. This context allows plugins to access information about the current state of the system.

## Security Considerations

The dynamic execution of plugin code using `exec()` introduces potential security risks. It's crucial to carefully validate and sanitize plugin code to prevent malicious code from being executed. Access control mechanisms should be implemented to restrict which users can create, modify, and activate plugins. *Always be cautious when installing plugins from untrusted sources.*

## Creating a Plugin

*(This section needs more content. We'll add more details later. It should describe the basic steps involved in creating a new plugin, such as creating a new file, defining the plugin class, and implementing the necessary methods.)*

## Installing a Plugin

*(This section needs more content. We'll add more details later. It should describe how to install a plugin, whether it's through a web interface, a command-line tool, or by manually copying files.)*

## Configuration: Valves

Plugins can define configuration options, often called "valves," to allow customization of their behavior. Valves are defined using Pydantic models within a nested class named `Valves` inside the main plugin class. These options can be adjusted globally or on a per-user basis.

```python
from pydantic import BaseModel, Field

class MyPluginType:  # Replace MyPluginType with Filter, Pipe, or Action
    class Valves(BaseModel):
        """
        Configuration parameters (valves) for the plugin.
        """
        parameter_1: str = Field(default="value1", description="Description of parameter 1")
        parameter_2: int = Field(default=10, description="Description of parameter 2", ge=0, le=100)
```

Valves provide a structured and type-safe way to manage plugin settings. Leverage Pydantic's validation features (e.g., `ge`, `le`, `...` for required fields) to ensure the integrity of valve configurations.  *How valve configurations are managed and accessed within the Open WebUI system needs further documentation. (Is there a specific UI for managing valves? How are valve settings stored?)*. Accessing valve values, if defined, is done via `self.valves.parameter_name`.

## Error Handling

*(This section needs more content. It should describe how plugins should handle errors gracefully and how errors are reported to the user or the system.)*

## API Endpoints

The Open WebUI Function system provides the following RESTful API endpoints for managing and interacting with functions:

| Endpoint                               | Method | Description                                                                     |
| :------------------------------------- | :----- | :------------------------------------------------------------------------------ |
| `/api/v1/functions`                     | GET    | List all registered functions.                                                 |
| `/api/v1/functions/create`              | POST   | Create a new function.                                                          |
| `/api/v1/functions/id/{id}`             | GET    | Retrieve a specific function by ID.                                             |
| `/api/v1/functions/id/{id}`             | PUT    | Update an existing function.                                                    |
| `/api/v1/functions/id/{id}`             | DELETE | Delete a function.                                                              |
| `/api/v1/functions/id/{id}/valves`      | GET    | Get the valve configuration for a function.                                    |
| `/api/v1/functions/id/{id}/valves/update` | POST   | Update the valve configuration for a function.                                 |
| `/api/v1/functions/{function_id}.{model_id}` | POST   | Execute a specific model of a function (e.g., `/api/v1/functions/my_function.model_1`). |
# Open WebUI Functions

This repository contains a collection of helper scripts and enhancement functions designed to extend the capabilities of Open WebUI. These tools streamline development and provide advanced features, including integration with Google's Gemini API.

## `gemini_manifold.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/gemini_manifold)

### Description

This is a manifold pipe function that adds support for Google's Gemini Studio API.

### Features

-   Whitelist based model retrieval and registration.
-   Support for text and image input.
-   Streaming and non-streaming content generation.
-   ~~Special handling for thinking models, including thought encapsulation.~~  
    API stopped providing thoughts https://github.com/googleapis/python-genai/issues/226 :(
-   Support for [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/grounding?lang=python)

### Usage

Requires Open WebUI v0.5.5 or later.

## Additional Scripts

-   `thinking_gemini.py`: [Open WebUI Community](https://openwebui.com/f/suurt8ll/thinking_gemini) An outdated script, initially developed for `gemini-2.0-flash-thinking-exp` model interactions. Superseded by `gemini_manifold.py`.
-   `function_updater.py`: Automates updating functions on a server via a REST API, streamlining development.
-   `smallthinker_filter.py`: At first I wanted to code a simple filter that makes output of the `smallthinker:3b` nicer but now I'm thinking of turning it into more general reasoning model formatter.
-   `system_prompt_injector.py`: The idea is to allow chaning chat options like system prompt and temperature from the chatbox. It would pair nicely with Prompts feature Open WebUI offers.
-   `venice_manifold.py`: [Open WebUI Community](https://openwebui.com/f/suurt8ll/venice_image_generation) enables image creation by using any diffusion model offered by Venice.ai API.
-   `pipe_function_template.py`: Helpful skeletion (template) file for speeding up creation of new `Pipe` functions.

## Contributing

Contributions are welcome. Fork the repository, create a feature branch, commit your changes, and open a pull request.

## License

MIT License. See the `LICENSE` file for details.

## Acknowledgements

Thanks to the Open WebUI team and all contributors. Special thanks to Matthew for the groundwork on `thinking_gemini.py`.

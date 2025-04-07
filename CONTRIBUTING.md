# Contributing to Open WebUI Functions

Thank you for considering contributing to this project! Your contributions are greatly appreciated.

## How to Contribute

Here's how you can contribute to this project:

1.  **Fork the repository:** Click the "Fork" button at the top right of the repository page on GitHub. This will create a copy of the repository under your GitHub account.

2.  **Clone your fork:** Clone the forked repository to your local machine:

    ```bash
    git clone https://github.com/YOUR_USERNAME/suurt8ll-open_webui_functions.git
    cd suurt8ll-open_webui_functions
    ```

3.  **Create a feature branch:** Create a new branch for your changes. Use a descriptive name for your branch, such as `feature/new-plugin` or `fix/bug-description`:

    ```bash
    git checkout -b feature/your-feature-name
    ```

4.  **Make your changes:** Implement your changes, following the project's coding conventions and style. Be sure to test your changes thoroughly.

5.  **Commit your changes:** Commit your changes with clear and concise commit messages:

    ```bash
    git add .
    git commit -m "Add: New feature or Fix: Bug description"
    ```

6.  **Push your changes:** Push your branch to your forked repository on GitHub:

    ```bash
    git push origin feature/your-feature-name
    ```

7.  **Create a pull request:** Go to your forked repository on GitHub and click the "Create Pull Request" button.  Make sure the pull request is targeted at the `master` branch of the main repository.

## Development Environment

The `dev/` directory offers a convenient environment for developing and testing plugins quickly. It includes:

*   `function_updater.py`: A script that automatically updates functions in your Open WebUI instance when you modify the plugin code.
*   `dev.sh`: A script to set up the development environment.
*   `.env.example`: An example environment file for configuring settings.

**Note:** The `dev/` environment is currently tailored for a Linux-based development setup. Contributions to improve cross-platform compatibility are welcome!

## Coding Conventions

*   Follow PEP 8 style guidelines for Python code.
*   Write clear and concise code with meaningful variable and function names.
*   Document your code with docstrings and comments.
*   Adhere to the Open WebUI plugin system conventions (e.g., naming plugin classes `Filter`, `Pipe`, or `Action`).

## Testing

*   Test your changes thoroughly before submitting a pull request.
*   Consider adding unit tests to ensure the stability and reliability of your code.

## Code Review

*   All pull requests will be reviewed by project maintainers.
*   Be prepared to address any feedback or suggestions provided during the review process.

## Thank You!

Thank you for your interest in contributing to this project. Your contributions help make Open WebUI even better!
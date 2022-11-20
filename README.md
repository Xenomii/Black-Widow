# Introduction
This repository contains code for ICT3204 Security Analytics Coursework 2.

# Requirements
To run this project, you will need to install [Python 3.9.15](https://www.python.org/downloads/release/python-3915/). 

This project uses [pyenv](https://github.com/pyenv/pyenv) to install and manage different versions of Python. It is highly recommended to use this tool to manage different versions of Python on your machine.

This project also uses [Poetry](https://python-poetry.org/) to install and manage the various Python packages.

## Pyenv

### Windows (WSL)
Run the command below to install some dependencies required by pyenv:
```
sudo apt-get install -y git gcc make openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev zlib1g-dev libncursesw5-dev libgdbm-dev libc6-dev zlib1g-dev libsqlite3-dev tk-dev libffi-dev libxcb*-dev libfontconfig1-dev libxkbcommon-x11-dev libgtk-3-dev libegl1 x11-apps
```
Execute the command below to install pyenv onto your machine:
```
curl https://pyenv.run | bash
```

<strong>You should get a warning stating that you have not added pyenv to the load path.</strong>

To do this, open up your shell file (e.g., .bashrc, .zsh, etc.) in your preferred editor:
```
sudo nano ~/.bashrc
```

Copy and paste the following text at the end of your shell file:
```
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Once that is done, restart your shell:
```
exec $SHELL
```

Run this command to ensure that you have installed pyenv correctly:
```
pyenv update
```

### macOS
Run these commands to install the dependencies that pyenv needs:
```
brew install readline xz
xcode-select --install
brew install openssl
```

To install pyenv, it uses the same command as Ubuntu:
```
curl https://pyenv.run | bash
```

Similar to Ubuntu, you would need to load pyenv onto the path by editing your shell file:
```
sudo nano ~/.bashrc
```

Copy the text below and add it to the end of the file:
```
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Restart your shell for the changes to take effect:
```
source ~/.bashrc
```

Ensure pyenv is installed correctly by running this command:
```
pyenv update
```

## Python
Once you have installed pyenv, you can begin using it to install and manage the specified version of Python required for this project.

To install the required version of Python using pyenv, run the command below:
```
pyenv install 3.9.15
```

To ensure that you are using that version of Python, run this command:
```
pyenv global 3.9.15
```

## Poetry
A tool for dependency management and packaging in Python. You can declare the libraries your project depends on and it will manage them for you.

To install it onto your machine, run this command:
```
curl -sSL https://install.python-poetry.org | python3 -
```

Proceed to add Poetry to your PATH by editing your shell file:
```
sudo nano ~/.bashrc
```

Copy and paste the following into your shell file:
```
export PATH="$HOME/.local/bin:$PATH"
```

After that, restart your shell:
```
exec $SHELL
```

Test that Poetry is working by running this command:
```
poetry --version
```

Before installing the packages, Poetry needs to be configured to use the correct version of Python:
```
poetry config virtualenvs.prefer-active-python true
```

Once this is completed, install the packages using Poetry:
```
poetry install
```
This will auto-create a virtual environment that will contain the installed packages based on `pyproject.toml`.

To access this virtual environment, simply run this command:
```
poetry shell
```

This command however, is quite buggy at times and an alternative for this command is to run this:
```
source $(poetry env info --path)/bin/activate
```

# Running the Project
Before you can run the project, you must first clone the repository and install the required Python dependencies.

## Clone GitHub Repository
Ensure that you select an appropriate directory to place the project into before cloning:
```
cd /path/to/project/directory
git clone https://github.com/ICT3204/Coursework-2
```

## Dependencies
Move into the cloned repository and install the required dependencies:
```
cd Coursework-2
poetry install
```
<i><strong>Ensure Poetry has been installed and configured correctly.</strong></i>

## Desktop Application
Once that is done, you can run the desktop application by executing the main Python file:
```
python3 src/app.py
```

# Development
This section lists some common development issues and tips for the project.

## Qt Designer
### 1. What is the command to open Qt Designer?
```
qt6-tools designer
```
### 2. Unable to open Qt Designer from WSL
Try to set this configuration option and see if it helps:
```
export QT_QPA_PLATFORM=xcb
```
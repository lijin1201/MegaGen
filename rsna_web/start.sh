#!/bin/bash

VENV_PATH=".venv"

# Check if the virtual environment is already activated
if [[ "$VIRTUAL_ENV" != *aienv ]]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "Virtual environment already active: $VIRTUAL_ENV"
fi

which pip
until python rsna_web.py; do
    echo "'rsna_webt.py' crashed with exit code $?. Restarting..." >&2
    sleep 1
done

#docker:
#echo docker run -d -e GRADIO_SERVER_PORT=1201 -p 1201:1201 --gpus all -w /app -v .:/app -v $(readlink -f assets):/app/assets\
# grapp python /app/rsna_web.py

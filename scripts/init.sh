eval "$(conda shell.bash hook)"
conda create -n chess python=3.10 poetry --y
conda activate chess
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install

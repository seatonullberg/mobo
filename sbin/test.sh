#!/bin/bash

mobo_dir="$(pwd)"

# static type checking
echo -e "\u001b[33m[mobo] mypy static type checking...\u001b[0m"
export MYPYPATH="$mobo_dir/mypy"
mypy --config-file "$MYPYPATH/mypy.ini" "$mobo_dir/mobo/"

# linting
echo -e "\u001b[33m[mobo] pyflakes linting...\u001b[0m"
pyflakes "$mobo_dir/mobo"

# unit testing
echo -e "\u001b[33m[mobo] pytest unit testing...\u001b[0m"
python3 -m pytest -vv "$mobo_dir/mobo/"

# cleanup
echo -e "\u001b[33m[mobo] removing generated files...\u001b[0m"
bash "$mobo_dir/sbin/cleanup.sh"
echo -e "\u001b[33m[mobo] test complete.\u001b[0m"
exit 0

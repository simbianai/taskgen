#!/bin/bash -e

rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install wheel
pip install toml-cli

VERSION=$(toml get --toml-path pyproject.toml project.version)

WHL_FILE=dist/taskgen_ai-${VERSION}-py3-none-any.whl
rm -f ${WHL_FILE}

python setup.py bdist_wheel

if [ ! -f ${WHL_FILE} ]; then
   echo "ERROR: ${WHL_FILE} was not generated"
   exit -1
fi

echo "-----------------------------------------------"
echo "File ${WHL_FILE} created"

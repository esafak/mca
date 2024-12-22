name: Run Tests on Pull Request

on:
  pull_request:
    branches:
      - main  # Change this to the branch you want to track for PRs, if needed

jobs:
  run-tests:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.13"  # Specify the Python version required for your project

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run test_mca.py
      run: |
        python tests/test_mca.py
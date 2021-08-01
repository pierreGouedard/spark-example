"""Builds the datalab package from the datalab folder.

To do so run the command below in the root folder:
pip install -e .
"""
from setuptools import setup, find_packages


# get the dependencies and installs
with open("requirements.txt", "r", encoding="utf-8") as f:
    # Make sure we strip all comments and options (e.g "--extra-index-url")
    # that arise from a modified pip.conf file that configure global options
    # when running kedro build-reqs
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)

setup(
    name="datalab",
    version="1.0",
    packages=find_packages(),
    author="Pierre Gouedard",
    author_email="pierre.gouedard@sia-partners.com",
    description="Package for custom data science module",
    install_requires=requires,

)

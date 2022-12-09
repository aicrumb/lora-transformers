import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="lora",
    py_modules=["lora"],
    version="0.0.1",
    description="Low Rank Adaptation for Transformers.",
    author="aicrumb",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)
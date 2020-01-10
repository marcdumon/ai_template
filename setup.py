# --------------------------------------------------------------------------------------------------------
# 2020/01/10
# src - setup.py.py
# md
# --------------------------------------------------------------------------------------------------------

import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="ai_template", # Replace with your own username
    version="0.0.1",
    author="Marc Dumon",
    author_email="dumon.marc@gmail.com",
    description="ML Template",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/marcdumon/ai_template",
    packages=setuptools.find_packages(),
    py_modules = ['machine','filter_visualisation','configuration'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
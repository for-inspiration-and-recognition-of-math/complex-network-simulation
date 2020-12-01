from setuptools import find_packages, setup

import os

# Optional project description in README.md:

current_directory = os.path.dirname(os.path.realpath(__file__)) 

try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = "Well, the detailed readme cannot be read. We'll proceed with installing dependencies"

setup(
    name="Simulating Non-zero-sum game on random networks", # Replace with your own username
    version="0.1.0",
    author="Group-G",
    author_email="",
    description="Simulation that outputs cool graphs. You can frame'em on a wall, or deck'em in your hall.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/for-inspiration-and-recognition-of-math/M168-simulation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
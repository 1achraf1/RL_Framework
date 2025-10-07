from setuptools import setup, find_packages

setup(
    name="rl-framework",              # package name
    version="0.1.0",                  # version
    author="Your Name",               # optional
    description="Educational RL framework with GridWorld environments",
    packages=find_packages(),         # finds rl_framework/
    install_requires=[
        "numpy",
        "matplotlib"
    ],
    python_requires=">=3.7",
)

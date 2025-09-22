#!/usr/bin/env python3
"""
Setup script for the Stress Monitoring System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stress-monitoring-system",
    version="1.0.0",
    author="AI Assistant",
    author_email="assistant@example.com",
    description="AI/ML-powered stress monitoring system with personalized recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stress-monitoring-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stress-monitor=run_app:main",
            "stress-train=train_model:main",
            "stress-test=test_system:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.html", "*.css", "*.js", "*.json"],
    },
    keywords="stress, monitoring, ai, ml, health, wellness, recommendations",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/stress-monitoring-system/issues",
        "Source": "https://github.com/yourusername/stress-monitoring-system",
        "Documentation": "https://github.com/yourusername/stress-monitoring-system#readme",
    },
)

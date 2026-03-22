"""Setup script for TRI-X-CDSS package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="trix-cdss",
    version="1.0.0",
    author="Chatchai Tritham, Chakkrit Snae Namahoot",
    author_email="chatchait66@nu.ac.th",
    description="TRI-X: Three-Tier Evaluation Framework for Dizziness Clinical Decision Support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChatchaiTritham/TRI-X-CDSS",
    project_urls={
        "Bug Tracker": "https://github.com/ChatchaiTritham/TRI-X-CDSS/issues",
        "Documentation": "https://github.com/ChatchaiTritham/TRI-X-CDSS/blob/main/README.md",
        "Source Code": "https://github.com/ChatchaiTritham/TRI-X-CDSS",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "viz": [
            "plotly>=5.17.0",
            "dash>=2.14.0",
            "kaleido>=0.2.1",  # Static image export
        ],
    },
    include_package_data=True,
    package_data={
        "trix_cdss": [
            "data/clinical_guidelines/*.json",
            "data/syndx/archetypes/*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "clinical decision support",
        "explainable AI",
        "digital twin",
        "causal inference",
        "multi-agent simulation",
        "dizziness",
        "vertigo",
        "medical AI",
        "healthcare",
    ],
)

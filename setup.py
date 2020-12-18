from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="chatspace",
    version="2.0.2",
    description="Spacing model for Korean chat-style texts",
    install_requires=["tensorflow>=2.0"],
    url="https://github.com/pingpong-ai/chatspace.git",
    author="ScatterLab",
    author_email="developers@scatterlab.co.kr",
    keywords=["spacing", "korean", "pingpong"],
    python_requires=">=3.6",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(exclude=["tests"]),
    package_data={
        "chatspace": [
            "resources/vocab",
            "resources/config.json",
            "resources/chatspace_model/*",
            "resources/chatspace_model/*/*",
        ]
    },
)

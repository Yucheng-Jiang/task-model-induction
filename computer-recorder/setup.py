from setuptools import find_packages, setup


setup(
    name="ComputerRecorder",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pillow",
        "mss",
        "pynput",
        "pyobjc-framework-Quartz",
        "python-dotenv>=1.0.0",
        "tqdm",
        "backports.tarfile",
    ],
    extras_require={
        "monitoring": ["psutil"],
    },
    entry_points={
        "console_scripts": [
            "crec=crec.cli:main",
        ],
    },
    description="Records macOS mouse, keyboard, and screen activity into a session trace",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
    ],
    # launcher.py and backend_lib use PEP 604 unions in type aliases, which are
    # evaluated eagerly and so fail on 3.9 even with __future__ annotations.
    python_requires=">=3.10",
)

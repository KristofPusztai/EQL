import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EQL-NN",
    version="2.0.1", # MAJOR.MINOR.MAINTENANCE
    author="Kristof Pusztai",
    author_email="kpusztai@berkeley.edu",
    description="A Tensorflow implementation of the Equation Learning Based Neural Network Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KristofPusztai/EQL",
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow>=2.11',
        'numpy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
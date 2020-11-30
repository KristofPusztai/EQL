import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EQL", # Replace with your own username
    version="0.0.5",
    author="Kristof Pusztai",
    author_email="kristofp12@gmail.com",
    description="An implementation of Equation Learning Based Neural Network Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KristofPusztai/EQL",
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_vis",
    version="0.0.1",
    author="Utku Ozbulak",
    description="A package to visualize pytorch CNN models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/utkuozbulak/pytorch-cnn-visualizations/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

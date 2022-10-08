from setuptools import setup


def read_file(path: str) -> str:
    with open(path, "r") as f:
        res = f.read()
    return res


description = "Mini-Lightning is a lightweight machine learning training library, which is a mini version of Pytorch-Lightning with only 1k lines of code. It has the advantages of faster, more concise and more flexible."
long_description = read_file("README.md")
install_requires = read_file("requirements.txt").splitlines(False)
classifiers = [
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    'Programming Language :: Python',
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]
setup(
    name="mini-lightning",
    version="0.1.4",
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT",
    url="https://github.com/ustcml/mini-lightning/",
    author="Jintao Huang",
    author_email="huangjintao@mail.ustc.edu.cn",
    packages=["mini_lightning"],
    install_requires=install_requires,
    classifiers=classifiers,
    python_requires=">=3.8"
)

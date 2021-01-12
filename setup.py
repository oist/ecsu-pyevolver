import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    reqs = f.read().strip().split('\n')

setuptools.setup(
    name="pyevolver",  # distribution name of the package; unique on pypi.org
    version="0.2.0",
    author="Federico Sangati, Katja Sangati",
    author_email="federico.sangati2@oist.jp, ekaterina.sangati@oist.jp",
    description="Package for evolving neural networks with Python.",
    long_description=long_description,  # loaded from README
    long_description_content_type="text/markdown",
    url="https://gitlab.com/oist-ecsu/pyevolver",  # where the package will be hosted
    # a list of all Python import packages that should be included in the distribution package
    # in this case it'll only be example_pkg
    # packages=setuptools.find_packages(exclude=['contrib', 'docs']),
    packages=['pyevolver', 'pyevolver.tests'],
    install_requires=reqs,
    classifiers=[
        "Programming Language :: Python :: 3.7.3",  # compatible only with Python 3.7.3
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

# RUN: python setup.py sdist bdist_wheel

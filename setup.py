from setuptools import find_packages, setup

setup(
    name='leodecay',
    version='1.0.0',
    description='Tools for downloading, preprocessing, and analyzing space weather and LEO satellite orbital decay data.',
    author='Vanessa Mercea',
    author_email='vanessa-maria.mercea@unibe.ch',
    license='MIT',
    packages=find_packages(include=["leodecay*"]),
    python_requires='>=3.9',
)
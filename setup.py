#!/usr/bin/env python3

from setuptools import setup, find_packages
from os import path
import versioneer

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='deepracer-utils',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(include=["deepracer", "deepracer.*"]),
    description='A set of tools for working with DeepRacer training',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aws-deepracer-community/deepracer-utils/',
    author='AWS DeepRacer Community',
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT No Attribution License (MIT-0)',

        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Internet :: Log Analysis'
    ],
    keywords='aws deepracer awsdeepracer',
    python_requires='>=3.8,<4.0',
    install_requires=[
        'boto3>=1.12.0',
        'python-dateutil<3.0.0,>=2.1',
        'numpy>=1.18.0',
        'shapely>=1.7.0',
        'matplotlib>=3.1.0',
        'pandas>=1.0.0',
        'scikit-learn>=0.22.0',
        'joblib>=0.17.0'
    ],
    extras_require={
        'visualization': ['tensorflow', 'opencv-python', 'python-resize-image'],
        'dev': ['check-manifest'],
        'test': ['tensorflow','coverage','opencv-python','python-resize-image'],
    },
    project_urls={
        'Bug Reports':
        'https://github.com/aws-deepracer-community/deepracer-utils/issues',
        'Source':
        'https://github.com/aws-deepracer-community/deepracer-utils/',
    },
    include_package_data=True,
)

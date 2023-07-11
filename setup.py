#!/usr/bin/env python3

"""Setup script and metadata to show on PyPi web"""

from setuptools import setup

with open('README.md', 'r', encoding = 'utf-8') as readme_file:
    readme = readme_file.read()

requirements = [
    'opencv-python',
    'numpy',
]

test_requirements = [
    'pylint',
    'pytest',
]

setup(
    name='imgdups',
    version='0.1.5',
    description="Very fast two folder image duplicate finder programmed with pickle and cv2",
    long_description=readme,
    long_description_content_type = 'text/markdown',
    author="Chuck Norrison",
    author_email='itsmells@yourshorts.club',
    url='https://github.com/ChuckNorrison/imgdups/',
    packages=['imgdups'],
    package_dir={'imgdups': 'imgdups'},
    include_package_data=True,
    install_requires=requirements,
    license="GPLv3",
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'imgdups = imgdups.imgdups:main',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.6',
)

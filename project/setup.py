"""Setup."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2025(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 09 Apr 2025 10:36:34 AM CST
# ***
# ************************************************************************************/
#

from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="image3d",
    version="1.0.0",
    author="Dell Du",
    author_email="18588220928@163.com",
    description="Image 3D package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/delldu/Hunyuan3D-2.git",
    packages=["image3d"],
    package_data={"image3d": ["models/image3d.pth"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 1.9.0",
        "todos >= 1.0.0",
    ],
)

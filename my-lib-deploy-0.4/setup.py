from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README


setup(name="package_outlier",
      version="0.4",
      description="This is an outlier detection package",

      install_requires=["scipy", "numpy", "pandas"],


      long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/jainyk",
    author="Vishal Jain",
    author_email="vjiit97@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["package_outlier"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "weather-reporter=package_outlier:entry_point",
        ]
    }
      )

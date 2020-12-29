from setuptools import setup

def readme():
    with open('README.md', encoding="utf8") as f:
        README = f.read()
    return README


setup(name="package_outlier",
      version="0.9",
      description="This is an outlier detection package",

      install_requires=["scipy", "numpy", "pandas","scikit-learn","statistics","datetime"],


    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/jainyk/package-outlier",
    MAINTAINER="Vishal Jain",
    MAINTAINER_EMAIL="vjiit97@gmail.com",
    license="MIT",
    PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/jainyk/package-outlier/issues',
    'Documentation': 'https://github.com/jainyk/package-outlier/blob/main/README.md',
    'Source Code': 'https://github.com/jainyk/package-outlier/tree/main/my-lib-deploy-0.4'
},
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

from setuptools import find_packages, setup

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    zip_safe=True,
    include_package_data=True,
)

from setuptools import setup, find_namespace_packages


def get_requirements(kind=None):
    # Straight from bilby. May allow separate installation
    if kind is None:
        filename = "requirements.txt"
    else:
        filename = f"{kind}_requirements.txt"
    with open(filename, "r") as file:
        requirements = file.readlines()
    return requirements


setup(
    name='dtempest-core',
    packages=find_namespace_packages(where='src/', include=['dtempest.core', 'dtempest.core._pesum_deps']),
    package_dir={'': 'src'},
    version='0.2.0',
    description='Implementation of an NPE approach to gravitational wave parameter estimation',
    author='Daniel Lanchares',
    license='MIT',
    python_requires=">=3.10",
    install_requires=get_requirements(),
)

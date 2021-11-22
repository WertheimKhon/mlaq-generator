import setuptools

with open('README.md', 'r') as infile:
    long_description = infile.read()

setuptools.setup(
    name='generator',
    version='0.1.0',
    author='Christer Dreierstad',
    author_email='christerdr@outlook.com',
    description='Generative algorithm for an MD project using CNN to predict strength of materials',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chdre/generator',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=['torch',
                      'lammps_simulator @ git+https://git@github.com/evenmn/lammps-simulator@master#egg=lammps_simulator',
                      'run_torch_model @ git+https://git@github.com/chdre/run-torch-model@master#egg=run_torch_model',
                      'simplexgrid @ git+https://git@github.com/chdre/simplexgrid@master#egg=simplexgrid',
                      'data_analyzer @ git+https://git@github.com/chdre/data-analyzer@master#egg=data_analyzer',
                      'molecular_builder'],
    include_package_data=True,

)

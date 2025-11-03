from setuptools import setup, find_packages,setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        HYPEN_E_DOT = '-e .'

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='ml_project',
    version ='0.1.0',
    author='Shar',
    author_email= 'sharvarimashalkar21@gmail.com',
    find_packages = find_packages(),
    install_requires=get_requirements('requirements.txt')
    
    ) 
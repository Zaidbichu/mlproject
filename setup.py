from setuptools import find_packages,setup
from typing import list
def get_requirements(file_path:str)->list[str]:
    'this function will return the list of requirements'
    requirements=[]

    with open(file_path) as file_obj:
        requirements=file_obj.readline()
        requirements=[req.replace("\n"," ") for req in requirements]
    return requirements
setup(
    name='mlproject',
    version='0.0.1',
    author='zaid',
    author_email='zaidbichu4@gmail.com',
    packages=find_packages(),
    install_requirements=get_requirements('requirements.txt')
)
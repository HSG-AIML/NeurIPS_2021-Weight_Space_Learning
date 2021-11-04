from setuptools import setup

setup(
    name="versai",
    version="1.0.0",
    author="Konstantin Schürholt",
    author_email="konstantin.schuerholt@unisg.ch",
    packages=["model_definitions", "checkpoints_to_datasets"],
    description="Package to learn representations of Neural Network weights, i.e., replicate https://arxiv.org/abs/2110.15288",
)

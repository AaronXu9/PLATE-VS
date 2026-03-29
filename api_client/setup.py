from setuptools import setup

setup(
    name="platevs-client",
    version="0.1.0",
    description="API Client for PLATE-VS (Protein-Ligand Affinity & Target Evaluation)",
    author="PLATE-VS Team",
    py_modules=["platevs_client"],
    install_requires=[
        "requests>=2.28.0",
        "pandas>=1.5.0",
    ],
    python_requires=">=3.7",
)

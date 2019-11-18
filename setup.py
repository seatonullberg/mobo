import setuptools

with open("README.md", "r") as f:
      long_description = f.read()

setuptools.setup(
      name="mobo",
      version="1.0.0",
      author="Seaton Ullberg",
      author_email="sullberg@ufl.edu",
      description="A rational and extensible algorithm for solving multi-objective optimization problems",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/seatonullberg/mobo",
      packages=setuptools.find_packages(),
      license="BSD 2-Clause License"
)

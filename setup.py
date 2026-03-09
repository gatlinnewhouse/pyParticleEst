from setuptools import setup

name = "pyParticleEst"
version = "1.1.4"
packages = [
    "pyparticleest",
    "pyparticleest.models",
    "pyparticleest.paramest",
    "pyparticleest.utils",
]
url = "http://www.control.lth.se/Staff/JerkerNordh/pyparticleest.html"
author = "Jerker Nordh"
author_email = "ajn@ajn.se"
description = "Framework for particle based estimation methods, such as particle filtering and smoothing"
lic = "LGPL"

setup(
    name=name,
    version=version,
    packages=packages,
    url=url,
    author=author,
    author_email=author_email,
    description=description,
    license=lic,
)

from setuptools import setup

setup(
    name="skddp_se",
    version="0.0.1",
    install_requires=["torch", "torchaudio", "pyyaml", "tqdm", "soundfile"],
    extras_require={
        "develop": ["matplotlib"]
    },
    entry_points={}
)
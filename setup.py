# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for DeltaTTG: shared dependencies and bundled runtime resources.
# setup.py supplies dynamic dependency metadata to pyproject.toml.
import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "deltattg/version.txt")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "codetiming",
    "httpx",
    "json-repair",
    "fastapi",
    "uvicorn",
    "datasets",
    "dill",
    "hydra-core",
    "numpy",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0,<=2.50.0",
    "torchdata",
    "tensordict==0.10.0",
    "transformers==4.57.3",
    "wandb",
    "packaging>=20.0",
    "qwen-vl-utils[decord]",
    "gymnasium==0.29.1",
    "gym==0.24.0",
    "beautifulsoup4==4.11.1",
    "cleantext==1.1.4",
    "Flask==2.1.2",
    "Werkzeug==2.1.0",
    "rank_bm25==0.2.2",
    "rich>=13.7.1",
    "scikit-learn>=1.5",
    "thefuzz==0.19.0",
    "pyserini==0.17.0",
    "faiss-cpu>=1.8",
    "alfworld==0.4.2",
    "textworld==1.7.0",
]

GPU_REQUIRES = ["flash-attn==2.8.3"]
VLLM_REQUIRES = ["tensordict==0.10.0", "torch==2.8.0", "vllm==0.11.0"]
SGLANG_REQUIRES = [
    "tensordict==0.10.0",
    "sglang[srt,openai]==0.5.5",
    "torch==2.8.0",
]
TRL_REQUIRES = ["trl<=0.9.6"]
MCORE_REQUIRES = ["mbridge"]
TRANSFERQUEUE_REQUIRES = ["TransferQueue==0.1.2.dev0"]

extras_require = {
    "gpu": GPU_REQUIRES,
    "vllm": VLLM_REQUIRES,
    "sglang": SGLANG_REQUIRES,
    "trl": TRL_REQUIRES,
    "mcore": MCORE_REQUIRES,
    "transferqueue": TRANSFERQUEUE_REQUIRES,
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="deltattg",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    license="Apache 2.0",
    author="DeltaTTG contributors",
    description="DeltaTTG: reusable task progress meters for agent reinforcement learning",
    python_requires=">=3.11",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "deltattg": ["version.txt"],
        "verl": ["trainer/config/*.yaml"],
        "agent_system": ["environments/env_package/alfworld/configs/*.yaml", "environments/env_package/alfworld/alfworld/**/*"],
        "configs": ["actor/*.yaml", "actor/tasks/*.yaml", "actor/models/*.yaml", "actor/methods/*.yaml", "meter/*.yaml", "meter/tasks/*.yaml", "meter/models/*.yaml", "launch/*.yaml", "ablations/*.yaml"],
    },
    include_package_data=True,
    license_files=["LICENSE", "README.md", "licenses/*.txt"],
    long_description=long_description,
    long_description_content_type="text/markdown",
)

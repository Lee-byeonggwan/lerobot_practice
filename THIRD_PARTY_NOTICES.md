# Third-party sources

- **LeIsaac**, LightwheelAI, Apache-2.0: <https://github.com/LightwheelAI/leisaac>. Included material: `sim2sim/so101_isaac/patches/leisaac-isaaclab-v3.patch`. The patch changes quaternion/tensor APIs, task registration, scene selection and action configuration for the pinned IsaacLab version. Original copyright notices remain in upstream source. License text: [licenses/LeIsaac-Apache-2.0.txt](licenses/LeIsaac-Apache-2.0.txt).
- **IsaacLab**, Isaac Lab Project Developers: <https://github.com/isaac-sim/IsaacLab>. Downloaded at the documented commit; not vendored. See that checkout's license and dependency notices.
- **LeRobot**, Hugging Face: <https://github.com/huggingface/lerobot>. Installed as a dependency; not vendored or modified in site-packages by this repository. Runtime wrappers adapt the asynchronous observation pipeline.
- **SmolVLA base model**: <https://huggingface.co/lerobot/smolvla_base>. Downloaded separately. Its model card/license applies to model use.
- **Isaac Sim**, NVIDIA: <https://docs.isaacsim.omniverse.nvidia.com/>. Downloaded separately and subject to NVIDIA's license. Environment activation does not itself set license acceptance.
- **SO-101 USD and Table with Cube assets**, LightwheelAI: <https://huggingface.co/LightwheelAI/leisaac_env> and <https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.2>. Downloaded separately, hash-checked, and not distributed in this repository. Consult their upstream terms before redistributing assets.

This file records sources; it does not grant a blanket license over all project files. The repository owner has not yet selected a project-wide license for the experiment scripts.

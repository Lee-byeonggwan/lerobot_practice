# Usage: source sim2sim/so101_isaac/activate_sim.sh
sim_experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$sim_experiment_root/.venv-sim/bin/activate"
export LEISAAC_ASSETS_ROOT="$sim_experiment_root/assets"
export LEISAAC_TASK_SCOPE=lift_cube
export LEISAAC_LIFTCUBE_SCENE="$sim_experiment_root/assets/scenes/table_with_wood_cube_cup/collection_scene.usda"
# Set OMNI_KIT_ACCEPT_EULA=YES yourself after accepting NVIDIA's Isaac Sim license.
unset sim_experiment_root

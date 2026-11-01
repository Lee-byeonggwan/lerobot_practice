"""Small task configuration over the locally migrated official LeIsaac LiftCube."""
from pathlib import Path
from pxr import Gf
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.envs.mdp.observations import image_rgb
from isaaclab_visualizers.kit import KitVisualizerCfg
from leisaac.tasks.lift_cube.lift_cube_env_cfg import LiftCubeEnvCfg
from leisaac.tasks.template.single_arm_env_cfg import SingleArmTaskSceneCfg

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ('wrist', 'front', 'top')
CUBE_REFERENCE_XY = (.19, -.43)  # From Belly: 10 cm table-forward (+Y), 10 cm robot-right (+X).
CUBE_PLACEMENT_DESCRIPTION = 'From Belly lens: 10 cm table-forward (+Y), 10 cm robot-right (+X); horizontal distances'
OFFSETS = ((0,0),(.03,0),(-.03,0),(0,.03),(0,-.03))
INSTRUCTION = 'Pick up the wooden cube and place it inside the paper cup.'

def fixed_camera(name, eye, target):
    mat = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
    q = mat.ExtractRotationQuat()
    return TiledCameraCfg(
        prim_path='{ENV_REGEX_NS}/'+name+'_camera',
        offset=TiledCameraCfg.OffsetCfg(pos=eye,rot=(*q.GetImaginary(),q.GetReal()),convention='opengl'),
        data_types=['rgb'], width=640, height=480, update_period=1/30,
        background_color=(0,0,0),
        spawn=sim_utils.PinholeCameraCfg(focal_length=18, horizontal_aperture=36,
                                       clipping_range=(.01,50),lock_camera=True))

def base_mounted_camera(name, eye_world, target_world, focal_length):
    # Fixed base has 180-degree yaw and world origin (.09,-.60,.011).
    # Express both eye and target in that base frame so mounts follow the base.
    def local(p):
        return (.09-p[0],-.60-p[1],p[2]-.011)
    cam = fixed_camera(name,local(eye_world),local(target_world))
    cam.prim_path = '{ENV_REGEX_NS}/Robot/base/'+name+'_camera'
    cam.spawn.focal_length = focal_length
    return cam

def make_cfg(device):
    cfg = LiftCubeEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    cfg.sim.dt = 1/60
    cfg.decimation = 2
    cfg.sim.render_interval = 2
    cfg.recorders = None  # Explicit official EpisodeData/HDF5 writer in collection runner.
    cfg.use_teleop_device('so101leader')
    cfg.task_description = INSTRUCTION
    cfg.scene.robot.spawn.usd_path = str(ROOT/'assets/robots/so101_follower_purple.usda')
    cfg.scene.robot.init_state.pos = (.09,-.60,.011)
    # Reduce crushing force for the light wooden block; pilot grasp must validate it.
    cfg.scene.robot.actuators['sts3215-gripper'].effort_limit_sim = .3
    cfg.scene.wrist = SingleArmTaskSceneCfg().wrist.copy()
    cfg.scene.wrist.background_color = (0,0,0)
    # Match mount topology in the user's reference photo. Numeric poses and
    # optics remain provisional until compared against actual camera images.
    cfg.scene.front = base_mounted_camera('front',(.09,-.53,.115),(.17,-.48,.05),10)
    # User measured lens height about 60 cm above the .040911 m tabletop.
    # Pole is offset 7 cm sideways to avoid looking through the arm; XY/aim/FOV
    # remain estimates until actual Top images are supplied.
    cfg.scene.top = base_mounted_camera('top',(.16,-.64,.64091131225),(.17,-.45,.05),18)
    for name in CAMERAS:
        setattr(cfg.observations.policy,name,ObservationTermCfg(
            func=image_rgb,params={'sensor_cfg':SceneEntityCfg(name),'normalize':False}))
    cfg.observations.policy.enable_corruption = False
    for name in list(vars(cfg.terminations)):
        setattr(cfg.terminations,name,None)
    for name in list(vars(cfg.events)):
        if name.startswith('domain_randomize_'):
            setattr(cfg.events,name,None)
    cfg.viewer.eye = (.34,-1.35,.82)
    cfg.viewer.lookat = (.32,-.36,.13)
    cfg.sim.default_visualizer_cfg = KitVisualizerCfg(
        eye=cfg.viewer.eye,lookat=cfg.viewer.lookat,background_color=(0,0,0))
    return cfg

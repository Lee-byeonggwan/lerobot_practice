"""Generate the scene editor's original seed layer without interactive edits."""
import os
from pathlib import Path
from pxr import Gf, Usd, UsdGeom, UsdLux

def create_seed(root):
    layout = root/'assets/scenes/table_with_wood_cube_cup/editor_layout.usda'
    if layout.exists():
        raise FileExistsError(layout)
    stage = Usd.Stage.CreateNew(str(layout))
    stage.GetRootLayer().subLayerPaths = ['scene.usda']
    stage.SetDefaultPrim(stage.GetPrimAtPath('/world'))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    robot = stage.DefinePrim('/world/Robot', 'Xform')
    robot.GetReferences().AddReference(os.path.relpath(root/'assets/robots/so101_follower.usd', layout.parent))
    xf = UsdGeom.Xformable(robot)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(.35,-.64,.01))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0,0,0,1))
    UsdLux.DomeLight.Define(stage, '/world/EditorLight').CreateIntensityAttr(1000)
    camera = UsdGeom.Camera.Define(stage, '/world/EditorCamera')
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(.85,-1.05,.65), Gf.Vec3d(.36,-.35,.07), Gf.Vec3d(0,0,1)).GetInverse())
    camera.CreateFocalLengthAttr(24)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
    stage.GetRootLayer().Save()

if __name__ == '__main__':
    create_seed(Path(__file__).resolve().parents[1])

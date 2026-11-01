"""Create isolated physics assets from the approved layout, without editor robot duplication."""
import json
import hashlib
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'assets/scenes/table_with_wood_cube_cup'

def main():
    source = FOLDER / 'real_style_layout.usda'
    out = FOLDER / 'collection_scene.usda'
    robot_out = ROOT / 'assets/robots/so101_follower_purple.usda'
    if out.exists() or robot_out.exists():
        raise FileExistsError('Collection assets already exist; do not overwrite silently')
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    stage = Usd.Stage.CreateNew(str(out))
    stage.GetRootLayer().subLayerPaths = ['real_style_layout.usda']
    for name in ('Robot', 'EditorLight', 'EditorCamera'):
        stage.OverridePrim('/world/' + name).SetActive(False)
    stage.OverridePrim('/Render').SetActive(False)
    stage.SetDefaultPrim(stage.GetPrimAtPath('/world'))
    # From Belly (.09,-.53): 10 cm forward (+Y), then 10 cm right (+X).
    # Keep interactive edits intact in the original layout.
    stage.GetPrimAtPath('/world/cube').GetAttribute('xformOp:translate').Set(Gf.Vec3d(.19,-.43,.05191131225))
    stage.GetPrimAtPath('/world/cup').GetAttribute('xformOp:translate').Set(Gf.Vec3d(.25,-.60,.04191131225))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().Save()
    robot_stage = Usd.Stage.CreateNew(str(robot_out))
    robot_stage.GetRootLayer().subLayerPaths = ['so101_follower.usd']
    original = Usd.Stage.Open(str(robot_out.parent/'so101_follower.usd'))
    default = original.GetDefaultPrim().GetPath()
    robot_stage.SetDefaultPrim(robot_stage.GetPrimAtPath(default))
    material = robot_stage.GetPrimAtPath(default.AppendPath('Looks/material_a_3d_printed/Shader'))
    material.GetAttribute('inputs:diffuse_color_constant').Set(Gf.Vec3f(.22,.035,.48))
    robot_stage.GetRootLayer().Save()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    print(json.dumps({'scene':str(out),'robot':str(robot_out),'source_unchanged_sha256':original_hash}))

if __name__ == '__main__':
    main()

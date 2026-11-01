"""Author requested colors/placement in a separate layer over the saved editor layout."""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from prepare_editor_seed import create_seed


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root/'assets/scenes/table_with_wood_cube_cup'
    source = folder/'editor_layout.usda'
    output = folder/'real_style_layout.usda'
    if output.exists():
        raise FileExistsError(output)
    if not source.exists():
        create_seed(root)
    shutil.copy2(source, folder/('editor_layout.before_real_style_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.usda'))
    source_stage = Usd.Stage.Open(str(source))
    hashes = {layer.realPath: hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
              for layer in source_stage.GetUsedLayers() if layer.realPath}
    stage = Usd.Stage.CreateNew(str(output))
    # Use the clean scene: prior interactive edits moved individual cup walls.
    # Preserve the old editor file in its backup rather than carrying damaged geometry forward.
    stage.GetRootLayer().subLayerPaths = ['scene.usda']
    Sdf.CreatePrimInLayer(stage.GetRootLayer(),'/world')
    for path in ('/world/Robot','/world/EditorLight','/world/EditorCamera'):
        Sdf.CopySpec(source_stage.GetRootLayer(),path,stage.GetRootLayer(),path)
    stage.GetRootLayer().customLayerData = {'practice_background': 'black'}
    stage.SetDefaultPrim(stage.GetPrimAtPath('/world'))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    bounds = UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
    tabletop = bounds.ComputeWorldBound(stage.GetPrimAtPath('/world/counter_right_main_group')).ComputeAlignedRange().GetMax()[2]
    placements = {'Robot':(.09,-.60,.011), 'cup':(.25,-.60,tabletop+.001), 'cube':(.09,-.42,tabletop+.011)}
    for name, xyz in placements.items():
        stage.GetPrimAtPath('/world/'+name).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*xyz))
    # Keep the measured 2 cm block and upright, unscaled cup even if the prior editor was manipulated.
    stage.GetPrimAtPath('/world/cup').GetAttribute('xformOp:orient').Set(Gf.Quatf(1,0,0,0))
    UsdGeom.Xformable(stage.GetPrimAtPath('/world/cup')).AddScaleOp().Set(Gf.Vec3f(1))
    stage.GetPrimAtPath('/world/cube').GetAttribute('xformOp:scale').Set(Gf.Vec3d(2/3))
    white = UsdShade.Material.Define(stage,'/world/Looks/PracticeWhiteTable')
    shader = UsdShade.Shader.Define(stage,white.GetPath().AppendChild('Surface'))
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.8))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.75)
    white.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    table = stage.GetPrimAtPath('/world/counter_right_main_group/geometry_0')
    UsdShade.MaterialBindingAPI.Apply(table).Bind(white,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    color = stage.GetPrimAtPath('/world/Robot/Looks/material_a_3d_printed/Shader').GetAttribute('inputs:diffuse_color_constant')
    if not color:
        raise RuntimeError('Expected SO101 plastic material not found')
    color.Set(Gf.Vec3f(.22,.035,.48))
    # View from behind the robot so work-area right is also easy to see on screen.
    camera = stage.GetPrimAtPath('/world/EditorCamera')
    camera.GetAttribute('xformOp:transform').Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(.34,-1.35,.82),Gf.Vec3d(.32,-.36,.13),Gf.Vec3d(0,0,1)).GetInverse())
    camera.GetAttribute('focalLength').Set(18)
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'RenderProduct':
            prim.CreateAttribute('omni:rtx:background:source:type',Sdf.ValueTypeNames.Token).Set('color')
            prim.CreateAttribute('omni:rtx:background:source:color',Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0))
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(output))
    bounds = UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
    table_bounds = bounds.ComputeWorldBound(stage.GetPrimAtPath('/world/counter_right_main_group')).ComputeAlignedRange()
    boxes = {}
    for name,path in [('base','/world/Robot/base'),('cup','/world/cup'),('cube','/world/cube')]:
        box = bounds.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        boxes[name] = {'min_m':list(box.GetMin()),'max_m':list(box.GetMax()),'size_m':list(box.GetSize())}
        for axis in (0,1):
            assert box.GetMin()[axis] >= table_bounds.GetMin()[axis], (name,'outside table')
            assert box.GetMax()[axis] <= table_bounds.GetMax()[axis], (name,'outside table')
        assert box.GetMin()[2] >= tabletop-1e-5, (name,'below table')
    assert all(abs(v-.02)<1e-6 for v in boxes['cube']['size_m'])
    assert placements['cup'][0]>placements['Robot'][0] and placements['cup'][1]==placements['Robot'][1]
    assert placements['cube'][1]>placements['Robot'][1] and placements['cube'][0]==placements['Robot'][0]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report = {'layout':str(output),'source_layers_unchanged':hashes,'world_positions_m':placements,
              'tabletop_z_m':tabletop,'bounds':boxes,'background':'black','table':'white',
              'robot_plastic':'purple; motor materials unchanged','render_verified':False,
              'physics_verified':False,'real_camera_alignment_verified':False}
    output.with_suffix('.validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

"""Pose-based contact-tolerant task score, independent of the online diagnostic.

The cup inner radius is 24 to 34 mm, and its floor top is at 1 mm.
Contact with these surfaces is allowed, with 0.5 mm solver tolerance.
A one-second window must remain contained and released with cube translation
span <=2 mm and orientation variation <=5 degrees. This avoids interpreting
constraint solver residual velocities as actual motion of a resting cube.
"""
import argparse
import json
import math
from pathlib import Path

def tip_position(joints, kin):
    def multiply(a,b):
        return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    transform=kin['base']
    for q,(a,b) in zip(joints,kin['frames']):
        c,s=math.cos(q),math.sin(q)
        r=[[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]]
        transform=multiply(multiply(multiply(transform,a),r),b)
    return [sum(transform[i][j]*kin['tip'][j] for j in range(3))+transform[i][3] for i in range(3)]

def evaluate(trace, kin=None):
    # Final task success requires physical separation. Closing the fingers again
    # after retreating does not undo a successful placement.
    trace=trace[-31:]
    good=[]
    separations=[]
    for r in trace:
        contained=all(.0005<=z<=.0695 and math.hypot(x,y)<=.024+.010*z/.07+.0005
                      for x,y,z in r['local_corners'])
        released=r['released']
        if kin is not None:
            tip=tip_position(r['state'][:5],kin)
            separation=math.sqrt(sum((x-y)**2 for x,y in zip(tip,r['cube'])))
            separations.append(separation)
            released=separation>.04
        good.append(contained and r['upright'] and released and r['lifted'] and r['cup_near'])
    # Validate the final one second; a transient earlier placement is insufficient.
    window=trace[-31:]
    if len(window)<31 or not all(good[-31:]):
        return {'success':False,'reason':'final one second not fully contained, released, upright and lifted'}
    def distance(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
    span=math.sqrt(sum((max(r['cube'][i] for r in window)-min(r['cube'][i] for r in window))**2 for i in range(3)))
    cup_span=math.sqrt(sum((max(r['cup'][i] for r in window)-min(r['cup'][i] for r in window))**2 for i in range(3)))
    q0=window[0]['cube_quat_xyzw'];norm0=math.sqrt(sum(x*x for x in q0))
    angular=max(2*math.acos(min(1.,abs(sum(x*y for x,y in zip(q0,r['cube_quat_xyzw'])))/
                (norm0*math.sqrt(sum(x*x for x in r['cube_quat_xyzw']))))) for r in window)
    success=span<=.002 and cup_span<=.002 and angular<=math.radians(5)
    return {'success':success,'reason':'cube released and stationary inside cup for final second' if success else 'placement still moving',
            'last_second_cube_span_m':span,'last_second_cup_span_m':cup_span,'last_second_orientation_span_rad':angular,
            'minimum_gripper_separation_m':min(separations) if separations else None}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args()
    kin=json.loads((a.folder/'scoring_kinematics.json').read_text())
    rows=[]
    for f in sorted(a.folder.glob('episode_*/result.json')):
        result=json.loads(f.read_text());trace=json.loads((f.parent/'trace.json').read_text())
        rows.append({'episode_index':result['episode_index'],'online_diagnostic_success':result['success'],**evaluate(trace,kin)})
    print(json.dumps({'completed':len(rows),'successes':sum(r['success'] for r in rows),'episodes':rows},indent=2))

if __name__=='__main__':main()

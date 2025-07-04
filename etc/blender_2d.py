# csv 파일을 blender의 scripting 창에서 입력하면 2D 스켈레톤 생성

import bpy, csv, re
from mathutils import Vector

# ===== 사용자 설정 =====
csv_path  = r"C:\Users\human\Desktop\kpop_project\data-files\processed\keypoints_all.csv"  # CSV 경로
SCALE     = 5.0    # [0–1] → Blender 단위 변환
Z_PLANE   = 0.0    # 바닥에서 Z 높이
RADIUS    = 0.02   # 스피어 반지름

# MediaPipe 33 랜드마크 이름 (디버깅용)
JOINT_NAMES = [f"J_{i}" for i in range(33)]
# GHUM 3D 공식 연결
SKELETON = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),
    (11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]

# x,y→Blender X,Z 평면 배치
def to_blender_xy(x,y):
    X = (x - 0.5) * SCALE
    Z = (y - 0.5) * SCALE
    return Vector((X, Z_PLANE, Z))

# 1) CSV 읽기
all_data = []
with open(csv_path, newline='') as cf:
    rd = csv.DictReader(cf)
    for r in rd:
        m = re.search(r'\d+', r['frame'])
        frame = int(m.group()) if m else 0
        pts = [(float(r[f'x{i}']), float(r[f'y{i}'])) for i in range(33)]
        all_data.append({'frame':frame, 'pts':pts})
all_data.sort(key=lambda e: e['frame'])
frames = [e['frame'] for e in all_data]

# 2) 씬 초기화
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
scene = bpy.context.scene
scene.frame_start, scene.frame_end = frames[0], frames[-1]

# 3) 스피어 생성
joint_objs = []
for i in range(33):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=RADIUS, location=(0,0,0))
    s = bpy.context.object
    s.name = JOINT_NAMES[i]
    for p in s.data.polygons:
        p.use_smooth = True
    joint_objs.append(s)

# 4) Curve(폴리라인) 생성: one spline per link
curve_data = bpy.data.curves.new("SkeletonCurve", type='CURVE')
curve_data.dimensions = '3D'
for idx,(s,e) in enumerate(SKELETON):
    sp = curve_data.splines.new('POLY')
    sp.points.add(1)  # 총 2점
    # 초기 위치(0,0,0,1)
    for p in sp.points:
        p.co = (0,0,0,1)
obj_curve = bpy.data.objects.new("SkeletonLines", curve_data)
bpy.context.collection.objects.link(obj_curve)

# 5) Animate: 스피어 위치 + Curve update
for entry in all_data:
    f = entry['frame']
    scene.frame_set(f)
    # 5-1) 스피어 위치
    for i,(x,y) in enumerate(entry['pts']):
        p3 = to_blender_xy(x,y)
        joint_objs[i].location = p3
        joint_objs[i].keyframe_insert("location", frame=f)
    # 5-2) Curve 각 스플라인 점 업데이트
    for idx,(s,e) in enumerate(SKELETON):
        sp = curve_data.splines[idx]
        p0 = to_blender_xy(*entry['pts'][s])
        p1 = to_blender_xy(*entry['pts'][e])
        sp.points[0].co = (p0.x, p0.y, p0.z, 1)
        sp.points[1].co = (p1.x, p1.y, p1.z, 1)
        # keyframe each point
        sp.points[0].keyframe_insert('co', frame=f)
        sp.points[1].keyframe_insert('co', frame=f)

print("✅ 2D→XZ Upright 애니메이션 + 선 연결 완료! 타임라인 ▶️ 확인해보세요.")

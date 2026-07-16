import json, os, sys
R=sys.argv[1]
raw=open(os.path.join(R,'design.json'),encoding='utf-8').read()
adir=os.path.join(R,'assets')
staged=sorted(os.listdir(adir)) if os.path.isdir(adir) else []
keep=[f for f in staged if f in raw]
drop=[f for f in staged if f not in raw]
print("KEEP (%d):"%len(keep))
for f in keep: print("  ",f)
print("DROP orphans (%d):"%len(drop))
for f in drop: print("  ",f)

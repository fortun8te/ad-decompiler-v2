import json, os, sys
R=sys.argv[1]
d=json.load(open(os.path.join(R,'design.json'),encoding='utf-8'))
refs=set()
def norm(p): return os.path.basename(str(p).replace('\\','/'))
def walk(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k in ('src','source','asset','asset_path','assetPath') and isinstance(v,str):
                refs.add(norm(v))
            walk(v)
    elif isinstance(o,list):
        for x in o: walk(x)
walk(d)
adir=os.path.join(R,'assets')
staged=set(os.listdir(adir)) if os.path.isdir(adir) else set()
print("REFERENCED (%d):"%len(refs))
for r in sorted(refs): print("  ",r)
print("STAGED but NOT referenced (%d):"%len(staged-refs))
for x in sorted(staged-refs): print("  ",x)
print("REFERENCED but MISSING (%d):"%len(refs-staged))
for x in sorted(refs-staged): print("  ",x)

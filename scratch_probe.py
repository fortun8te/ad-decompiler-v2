import json, os, sys
from PIL import Image
R=sys.argv[1]
d=json.load(open(os.path.join(R,'design.json'),encoding='utf-8'))
def walk(layers):
    for l in layers or []:
        if not isinstance(l,dict): continue
        m=l.get('mask')
        if isinstance(m,dict) and str(m.get('kind','')).lower()=='alpha':
            src=l.get('src') or ''
            meta=l.get('meta') or {}
            p=os.path.join(R, src.replace('\\','/'))
            try:
                im=Image.open(p); mode=im.mode
                has_alpha = mode in ('RGBA','LA','PA')
                extrema = im.getchannel('A').getextrema() if has_alpha else None
            except Exception as e:
                mode='ERR:%s'%e; extrema=None
            print(l.get('id'), '|', l.get('name'),'| mode=',mode,'| alpha_extrema=',extrema,'| mask_prov=',meta.get('mask_provenance'),'| own_cutout=',meta.get('ownership_cutout'))
        walk(l.get('children') or l.get('layers'))
walk(d.get('layers') or d.get('children'))

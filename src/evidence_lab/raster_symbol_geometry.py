"""Original-pixel table grid / anonymous legend-symbol shadow measurement.

Every cell and prototype is a real rectangle in a frozen source PNG. No OCR,
model call, answer, label, reference page or semantic rating is supplied here.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

import numpy as np
from PIL import Image

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def runs(indices):
    groups=[]
    for value in map(int,indices):
        if not groups or value>groups[-1][-1]+1: groups.append([value])
        else: groups[-1].append(value)
    return groups


def components(mask):
    remaining=mask.copy();height,width=mask.shape;result=[]
    for y,x in zip(*np.nonzero(mask)):
        y=int(y);x=int(x)
        if not remaining[y,x]: continue
        todo=deque([(y,x)]);remaining[y,x]=False;points=[]
        while todo:
            cy,cx=todo.popleft();points.append((cy,cx))
            for ny in range(max(0,cy-1),min(height,cy+2)):
                for nx in range(max(0,cx-1),min(width,cx+2)):
                    if remaining[ny,nx]:
                        remaining[ny,nx]=False;todo.append((ny,nx))
        result.append({'box':[min(p[1] for p in points),min(p[0] for p in points),
                              max(p[1] for p in points)+1,max(p[0] for p in points)+1],
                       'pixels':len(points)})
    return result


def pattern(mask):
    points=np.nonzero(mask)
    if len(points[0])<4: return None
    box=[int(points[1].min()),int(points[0].min()),int(points[1].max())+1,int(points[0].max())+1]
    tight=mask[box[1]:box[3],box[0]:box[2]]
    scaled=np.asarray(Image.fromarray(tight.astype('uint8')*255).resize((16,16),Image.Resampling.NEAREST))>0
    return {'box':box,'mask':scaled,'topology':len(components(tight)),
            'mask_sha256':hashlib.sha256(tight.tobytes()).hexdigest(),
            'normalized_mask_sha256':hashlib.sha256(scaled.tobytes()).hexdigest()}


def similarity(a,b):
    if a['topology']!=b['topology']: return 0.0
    left=np.pad(a['mask'],1);right=np.pad(b['mask'],1);best=0.
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            shifted=np.roll(right,(dy,dx),(0,1))
            common=np.logical_and(left,shifted).sum()
            best=max(best,float(2*common/(left.sum()+shifted.sum())))
    return best


def grid_and_cells(mask):
    height,width=mask.shape
    horizontal=runs(np.nonzero(mask.sum(axis=1)>=.55*width)[0])
    if not 4<=len(horizontal)<=40: return None,'no_bounded_drawn_grid'
    top=horizontal[0][0];bottom=horizontal[-1][-1]+1
    if bottom-top<40: return None,'drawn_grid_too_short'
    vertical=runs(np.nonzero(mask[top:bottom].sum(axis=0)>=.85*(bottom-top))[0])
    if not 3<=len(vertical)<=20: return None,'no_crossing_column_rules'
    cells=[]
    for row,(upper,lower) in enumerate(zip(horizontal,horizontal[1:])):
        for column,(left,right) in enumerate(zip(vertical,vertical[1:])):
            box=[left[-1]+3,upper[-1]+3,right[0]-2,lower[0]-2]
            if box[2]-box[0]<8 or box[3]-box[1]<8: continue
            p=pattern(mask[box[1]:box[3],box[0]:box[2]])
            if not p: continue
            glyph=p['box'];gw,gh=glyph[2]-glyph[0],glyph[3]-glyph[1]
            if (not 5<=gw<=min(32,.40*(right[0]-left[-1]))
                    or not 5<=gh<=min(32,.90*(lower[0]-upper[-1]))
                    or p['topology']>4): continue
            cellbox=[glyph[0]+box[0],glyph[1]+box[1],glyph[2]+box[0],glyph[3]+box[1]]
            cells.append({'row_index':row,'column_index':column,'cell_box_pixels':box,
                          'glyph_box_pixels':cellbox,'pattern':p})
    active_rows={c['row_index'] for c in cells}
    if len(cells)<6 or len(active_rows)<2: return None,'no_repeated_single_symbol_cells'
    return {'horizontal_rule_pixel_bands':[[g[0],g[-1]+1] for g in horizontal],
            'vertical_rule_pixel_bands':[[g[0],g[-1]+1] for g in vertical],
            'table_box_pixels':[vertical[0][0],top,vertical[-1][-1]+1,bottom],
            'row_count':len(horizontal)-1,'column_count':len(vertical)-1,'cells':cells},None


def match_legend(mask,grid):
    height,width=mask.shape;left,top,right,bottom=grid['table_box_pixels']
    band_top=min(height,bottom+4);band_bottom=min(height,bottom+max(120,bottom-top))
    region=mask[band_top:band_bottom]
    groups=[]
    source_components=components(region)
    nested_components=[]
    for component in source_components:
        x0,y0,x1,y1=component['box'];gw,gh=x1-x0,y1-y0
        if not (5<=gw<=28 and 5<=gh<=28 and 8<=component['pixels']<=220): continue
        enclosing=next((other for other in source_components if other is not component
            and other['box'][0]<=x0 and other['box'][1]<=y0
            and other['box'][2]>=x1 and other['box'][3]>=y1
            and other['box']!=component['box']),None)
        if enclosing is not None:
            nested_components.append({'inner_box_pixels':[x0,y0+band_top,x1,y1+band_top],
                'enclosing_box_pixels':[enclosing['box'][0],enclosing['box'][1]+band_top,
                    enclosing['box'][2],enclosing['box'][3]+band_top]})
            continue  # A disconnected inner mark belongs to the outer glyph.
        box=[x0,y0+band_top,x1,y1+band_top]
        p=pattern(mask[box[1]:box[3],box[0]:box[2]])
        if p is None or p['topology']>4: continue
        center=(x0+x1)/2
        aligned=next((g for g in groups if abs(g['center_x']-center)<=3),None)
        item={'box_pixels':box,'pattern':p}
        if aligned is None: groups.append({'center_x':center,'items':[item]})
        else: aligned['items'].append(item)
    proposals=[]
    for group in groups:
        items=sorted(group['items'],key=lambda i:i['box_pixels'][1])
        if not 2<=len(items)<=5: continue
        if len({(i['box_pixels'][1]+i['box_pixels'][3])//2 for i in items})!=len(items): continue
        if min(items[i+1]['box_pixels'][1]-items[i]['box_pixels'][3] for i in range(len(items)-1))<3: continue
        matches=[];all_unique=True
        for cell in grid['cells']:
            scored=sorted([(similarity(cell['pattern'],item['pattern']),index) for index,item in enumerate(items)],reverse=True)
            best,index=scored[0];runner=scored[1][0]
            unique=best>=.85 and best-runner>=.08
            all_unique &= unique
            matches.append({'row_index':cell['row_index'],'column_index':cell['column_index'],
                'cell_box_pixels':cell['cell_box_pixels'],'glyph_box_pixels':cell['glyph_box_pixels'],
                'anonymous_legend_index':index if unique else None,
                'glyph_mask_sha256':cell['pattern']['mask_sha256'],
                'observed_glyph_foreground_components':cell['pattern']['topology'],
                'match_similarity':best,'runner_up_similarity':runner,'unique_match':unique})
        if all_unique and len({m['anonymous_legend_index'] for m in matches})==len(items):
            proposals.append({'legend_column_center_x':group['center_x'],
                'legend_prototypes':[{'anonymous_legend_index':i,'box_pixels':item['box_pixels'],
                    'mask_sha256':item['pattern']['mask_sha256'],'foreground_components':item['pattern']['topology'],
                    'label_text_not_decoded':True} for i,item in enumerate(items)],
                'cell_matches':matches,'minimum_match_similarity':min(m['match_similarity'] for m in matches),
                'semantic_legend_labels_verified':False,'row_and_column_labels_verified':False})
    proposals.sort(key=lambda p:(-p['minimum_match_similarity'],p['legend_column_center_x']))
    return proposals[0] if len(proposals)==1 else None,{
        'nearby_component_alignment_groups':len(groups),'complete_unique_legend_proposals':len(proposals),
        'nested_source_components_combined_into_whole_glyphs':nested_components}



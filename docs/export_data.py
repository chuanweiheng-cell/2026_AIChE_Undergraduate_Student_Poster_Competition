#!/usr/bin/env python3
'''Export validated SciPy BVP tables into typed-array data for GitHub Pages.

Run this script from the package root after updating either original .npz table.
It requires NumPy, but the published site itself does not require Python.
'''
import gzip
import json
from pathlib import Path
import numpy as np

SOURCE = Path(__file__).resolve().parent.parent / 'EN_Five_Slider_Smooth_Kit' / 'sliders'
OUTPUT = Path(__file__).resolve().parent


def make_binary(filename, fields):
    '''Pack named arrays with 8-byte field alignment and gzip compression.'''
    payload = bytearray()
    info = []
    for name, values, dtype in fields:
        values = np.ascontiguousarray(values, dtype=dtype)
        if len(payload) % 8:
            payload.extend(b'\x00' * (8 - len(payload) % 8))
        offset = len(payload)
        payload.extend(values.tobytes(order='C'))
        info.append({'name': name, 'type': np.dtype(dtype).str, 'length': int(values.size), 'offset': offset})
    output_file = OUTPUT / filename
    with gzip.open(output_file, 'wb', compresslevel=9) as f:
        f.write(payload)
    print(output_file.name, 'uncompressed:', len(payload), 'compressed:', output_file.stat().st_size)
    return {'file': filename, 'fields': info, 'uncompressed_bytes': len(payload)}


with np.load(SOURCE / 'en_dimensional_profiles_2d.npz', allow_pickle=False) as a:
    nd = a['x'].size
    dimensional = make_binary('dimensional.bin.gz', [
        ('log_da', a['log_da'], '<f8'), ('gamma', a['gamma'], '<f8'),
        ('offsets', a['offsets'], '<u4'), ('x', a['x'], '<f8'),
        ('cH', a['c'][0], '<f4'), ('cNa', a['c'][1], '<f4'),
        ('cOH', a['c'][2], '<f4'), ('peak', a['peak'], '<f8'),
    ])
    assert a['offsets'][-1] == nd
    dimensional['count'] = len(a['log_da']) * len(a['gamma'])
    dimensional['relative_diffusivities'] = a['relative_diffusivities'].tolist()
    dimensional['model_signature'] = str(a['signature'].item())

with np.load(SOURCE / 'en_da_profiles.npz', allow_pickle=False) as a:
    nd = a['x'].size
    da = make_binary('da.bin.gz', [
        ('log_da', a['log_da'], '<f8'), ('offsets', a['offsets'], '<u4'),
        ('x', a['x'], '<f8'),
        ('cH', a['concentrations'][0], '<f4'),
        ('cNa', a['concentrations'][1], '<f4'),
        ('cOH', a['concentrations'][2], '<f4'),
        ('cCl', a['concentrations'][3], '<f4'),
    ])
    assert a['offsets'][-1] == nd
    da['count'] = len(a['log_da'])
    da['salt_ratio'] = float(a['salt_ratio'])
    da['model_signature'] = str(a['signature'].item())

manifest = {'version': 1, 'dimensional': dimensional, 'da': da}
(OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
print('All exports complete.')

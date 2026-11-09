"""Start a new edit from the STANDARD (edits/hold_on, Dexter x Hold On v3, approved by the user).

  python edit/tools/new_edit.py NAME --song media/song.mp3 --section media/NAME_section.json [--pack dexter_mix_1440p]

Run edit/tools/sections.py first to get the section json. This writes edits/NAME/config.py with
the standard's structure: the song settings filled in, plus 22 slots with the standard's
lengths (in half-beats), transitions, overlays, punch-ins, grade and push. Each slot says what
kind of shot it wants. You fill in src (in-point), pack and x (face position) for every slot;
importing the config refuses to build while any slot is empty. Then retune grade and gain per
shot. The standard's values suit footage like the Dexter packs (bright shots ~1.0 gain, night
shots 1.3-1.45).

The transitions are Dexter-themed (knife, scope, plastic, polaroid). For another character, keep
the slot structure and swap the signature transitions for that character's own objects.
"""
import argparse, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STANDARD = os.path.join(ROOT, 'edits', 'hold_on', 'config.py')
ROLES = [
    'OPEN: strong, well-lit face, outdoors or sunlit, ideally looking up (the loop returns here)',
    'outdoor or greenery shot with a gesture or look to camera',
    'sunlit face close-up with movement in it; the knife thrusts across it on the loudest early snare',
    'face framed by something (car window, doorway, mirror)',
    'dark, intense close-up (anger, heat); exits through smoke and embers',
    'head turn or look up in a dark room',
    'character holding their signature object near camera; set out_opts px, py on the object (microscope dive)',
    'workplace or medium shot with a look; exits through the blood pour (mid-edit climax)',
    'aftermath look, bright and pale (hard cut into a punch-in)',
    'confrontation: aiming or a hard stare at camera (punch-in on a loud snare, spins out)',
    'upside-down or disorienting shot',
    'camera or photo moment (flash turns into a Polaroid)',
    'joyful public moment, big smile',
    'cold stare against a dark or stained wall',
    'celebration or crowd close-up (confetti curtain out)',
    'the darkest red-lit stare of the edit',
    'iconic image: a poster, the name, a signature pose (punch-in on a loud hit)',
    'coloured-light profile (green or blue), very dark',
    'big smile or laugh, bright (red flash with rings out, on the loudest kick)',
    'the light on the face turns red (or another dramatic light change)',
    'cold, blue, quiet shot (kill-room plastic drags across)',
    'END: still, warm, looking off, long hold (light leak and particles; the blood drop loops to OPEN)',
]
KEEP = ('beats', 'out', 'out_opts', 'fx', 'punch', 'z', 'cy', 'sat', 'rk', 'con', 'tint', 'gain', 'push')


def standard_shots():
    sys.path[:0] = [os.path.join(ROOT, 'edit'), os.path.dirname(STANDARD)]
    import importlib.util
    spec = importlib.util.spec_from_file_location('standard_cfg', STANDARD); m = importlib.util.module_from_spec(spec)
    cwd = os.getcwd(); os.chdir(os.path.dirname(STANDARD))
    try: spec.loader.exec_module(m)
    finally: os.chdir(cwd)
    return m.SHOTS, m.FIRST_CUT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--song', required=True)
    ap.add_argument('--section', required=True); ap.add_argument('--pack', default='dexter_mix_1440p'); a = ap.parse_args()
    sec = json.load(open(a.section)); shots, first = standard_shots()
    assert len(shots) == len(ROLES), 'standard changed: update ROLES in new_edit.py'
    half = sec['half']; nhalf = int(round(sec['length'] / half))
    used = first + sum(s.get('beats', 0) for s in shots[1:])
    if nhalf != 64: print('note: the section is %d half-beats, the standard uses 64; add or remove 2-half-beat slots so the lengths sum to %d.' % (nhalf, nhalf - 1))
    d = os.path.join(ROOT, 'edits', a.name); os.makedirs(d, exist_ok=True); out = os.path.join(d, 'config.py')
    if os.path.exists(out): sys.exit('%s already exists' % out)
    rows = []
    for i, (s, role) in enumerate(zip(shots, ROLES)):
        kv = ', '.join('%s=%r' % (k, s[k]) for k in KEEP if k in s and not (k == 'z' and s[k] == 1.0))
        rows.append(' # %2d %s\n dict(name=%r, src=None, pack=None, x=0.5, %s),' % (i, role, 'slot%02d' % i, kv))
    song = os.path.relpath(os.path.join(ROOT, a.song), d)
    txt = f'''"""{a.name}: built from the standard (edits/hold_on, Dexter x Hold On v3).

Song: {os.path.basename(a.song)}, section {sec['offset'] // 60:.0f}:{sec['offset'] % 60:06.3f}, {sec['bpm']} BPM, {sec['bars']} bars ({sec['length']:.2f} s).
Loudest half-beats: {sec['loudest']}. The standard's big transitions sit on #10, #26, #38, #46, #52;
move cuts if this song's loud hits fall elsewhere (keep cuts on beats).
Fill src, pack (file in media/scenepacks/) and x for every slot, then: bash ../../edit/build.sh . --draft
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
SCENEPACK = PK + '{a.pack}.mp4'
SONG = {song!r}
SONG_OFFSET = {sec['offset']}
GRID0, HALF = 0.0, {half}
FIRST_CUT = {first}
TAIL = {sec['length']} - ({used} * {half} + 0.077)   # video is exactly the section, so it loops on the downbeat
PRE, POST = 0.60, 0.50

SHOTS = [
''' + '\n'.join(rows) + '''
]
for s in SHOTS:
    s.setdefault('z', 1.0)
    if s.get('pack') is None: s.pop('pack')
    elif not s['pack'].endswith('.mp4'): s['pack'] = PK + s['pack'] + '.mp4'

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
'''
    open(out, 'w').write(txt); print('wrote', out)


if __name__ == '__main__':
    main()

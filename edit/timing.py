"""Turns the half-beat plan into cut times.

Hard cuts, blurs, whips and punch-ins sit exactly on the grid point. Wipes are placed slightly
late (SHIFT) so the motion starts on the beat and lands just after it, as in the reference.
"""
SHIFT = {'glass': 0.13, 'slidewipe': 0.035, 'blood': -0.038, 'iris': 0.077, 'confetti': 0.05, 'plastic': 0.05, 'emblem': 0.0, 'strike': 0.0, 'tunnel': 0.077}
LONG_HOLD = 6


def build_cuts(shots, grid0, half, first_cut, tail):
    empty = [('%d (%s)' % (i, s['name'])) for i, s in enumerate(shots) if s.get('src') is None]
    if empty: raise ValueError('choose a clip (src=...) for slot ' + ', '.join(empty))
    k = first_cut; cuts = [0.0]
    for i, s in enumerate(shots):
        if i > 0:
            b = s.get('beats')
            if b is None: raise ValueError('shot %d (%s) needs beats=' % (i, s['name']))
            if not 2 <= b <= LONG_HOLD: raise ValueError('shot %d (%s): beats=%d, use 2-4 (one 5-6 hold)' % (i, s['name'], b))
            k += b
        cuts.append(grid0 + k * half + SHIFT.get(s.get('out'), 0.0))
    holds = [s['name'] for s in shots[1:] if s.get('beats', 0) > 4]
    if len(holds) > 1: print('warning: more than one long hold:', holds)
    return cuts, cuts[-1] + tail

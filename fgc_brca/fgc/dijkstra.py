"""Algorithm 3.3 (FUZZY-GRAPHOIDAL-COVER), steps 1-3: multi-label fuzzy Dijkstra.

* eternal (permanent) labels   <->  `perm[v]`
* provisional labels           <->  `open_[v]` + heap
* 'non-governing' provisional labels of the same vertex are all kept (Step 2.3 branching),
  capped by `max_labels` eternal labels per vertex.
* Step 2.2 (choose the provisional label that governs all others) is realised by popping the
  label with the smallest ranking value from a heap; a popped label is discarded if an
  eternal label of the same vertex already governs it.
"""
from __future__ import annotations
import heapq
import itertools


class Label:
    __slots__ = ("tfn", "v", "parent", "eid", "dead")

    def __init__(self, tfn, v, parent=None, eid=-1):
        self.tfn, self.v, self.parent, self.eid, self.dead = tfn, v, parent, eid, False


def _on_path(label, w):
    while label is not None:
        if label.v == w:
            return True
        label = label.parent
    return False


def fuzzy_dijkstra(adj, source, ops, max_labels=4):
    """adj[v] = list of (w, tfn_length, edge_id).  Returns list of eternal Labels."""
    n = len(adj)
    perm = [[] for _ in range(n)]
    open_ = [[] for _ in range(n)]
    cnt = itertools.count()
    start = Label((0.0, 0.0, 0.0), source)
    heap = [(0.0, next(cnt), start)]
    open_[source].append(start)
    while heap:
        _, _, lab = heapq.heappop(heap)
        if lab.dead:
            continue
        v = lab.v
        if lab in open_[v]:
            open_[v].remove(lab)
        if any(ops.governs(p.tfn, lab.tfn) for p in perm[v]):
            continue
        if len(perm[v]) >= max_labels:
            continue
        perm[v].append(lab)                       # becomes an eternal label
        for (w, L, eid) in adj[v]:
            if _on_path(lab, w):
                continue
            new = (lab.tfn[0] + L[0], lab.tfn[1] + L[1], lab.tfn[2] + L[2])
            if any(ops.governs(p.tfn, new) for p in perm[w]):
                continue
            if any(ops.governs(o.tfn, new) for o in open_[w]):
                continue
            for o in open_[w]:                    # new label governs older provisional ones
                if ops.governs(new, o.tfn):
                    o.dead = True
            open_[w] = [o for o in open_[w] if not o.dead]
            nl = Label(new, w, lab, eid)
            open_[w].append(nl)
            heapq.heappush(heap, (ops.rank(new), next(cnt), nl))
    return [l for v in range(n) for l in perm[v]]


def label_to_path(label):
    """Return (vertex tuple, edge-id tuple) of the path represented by a label."""
    verts, eds = [], []
    while label is not None:
        verts.append(label.v)
        if label.parent is not None:
            eds.append(label.eid)
        label = label.parent
    verts.reverse()
    eds.reverse()
    return tuple(verts), tuple(eds)

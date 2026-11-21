"""Resource limits: the one overflow rule of every path that consumes a tallymap resource.

The spawn Queue is a ring of Q slots.  A forward call needs only its PENDING children to fit: a child that finds the
ring full is dropped (stats['children_dropped']).  A path that reads the spawn TREE back from the ring (the tree
adjoint's record, _reverse_spawn.tree_vjp; Records + spawn, whose count pass is that record, _api._records) needs
every child of the call in it: a call with more than Q children in all has reused a slot and lost a node of the tree,
even when it dropped none.  queue_lost is that rule, for both kinds of consumer.
"""


def queue_lost(stats, Q, record):
    """-> a traced bool: the spawn Queue of capacity Q lost a child for this consumer.  stats: a launch's stats
    vector (int32[len(_api.STATS)]).  record=False (a forward call): a child was dropped.  record=True (a path that
    reads the tree from the ring): a child was dropped or the call had more than Q children in all (a slot was
    reused)."""
    from ._api import ST  # (here: _api imports this module)
    bad = stats[ST["children_dropped"]] > 0
    if record:
        bad = bad | (stats[ST["spawned"]] > Q)
    return bad

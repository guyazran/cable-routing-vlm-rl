"""Hierarchical DLO routing: the algorithm under test (reconstruction of arXiv:2510.19268).

This package is the algorithm only. It never imports the simulator or the environment
package (``cablerouting``); tests/test_separation.py enforces that.

  interface    the only contract with the outside world: scene and cable-state types, the
               pick-and-place command, rendered views, and the World protocol
  geometry     planar frames and cable arc length
  state        task-relative quantities derived from a cable state (head metrics, route status)
  decision     the structured high-level decision and its syntactic validation
  skills       Insert (frozen RL policy, closed loop), Pull and Flatten (scripted), as pick-and-place commands
  rl           insertion-policy observation, Eq. 1-4 reward, policies, training MDP and SAC
  planner      VLM planner: prompts and memory policy, serving client with failure classes
  controllers  VLM controller (+ true-progress diagnostic) and the ground-truth state reference controller
  agent        a high-level controller plus the shared skill library

The Isaac Sim environment implements :class:`dloroute.interface.World` in ``cablerouting.bridge``.
A physical robot would implement the same interface with perception and a motion stack.
"""

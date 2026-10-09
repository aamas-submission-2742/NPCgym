% Adapted from OFTEN-DeepRL pacman_program.lp, revision 3546dfd846f98e7f542b6e6107454bd5e2a5539c.
% Copyright (c) 2025 Lopez Miguel, Ignacio David. MIT; see THIRD_PARTY_NOTICES.md.
% Keep the reference's disjunctive player/ghost model and saturation check.
% NPC Gym supplies optimization and action IDs; horizon here includes time zero.

%%% Windowing %%%

#external gcol(G,C,0) : G=0..ghosts-1, C=-radius..radius.
#external grow(G,R,0) : G=0..ghosts-1, R=-radius..radius.
#external goutside(G) : G=0..ghosts-1.
#external wall(C,R) : C=-radius..radius, R=-radius..radius.


% player always starts at the 0 position
pcol(0,0).
prow(0,0).


%%% P_gen %%%

% movement of the agent
pmove(0, T) | pmove(1, T) :- T=1..horizon - 1.
prow(R + 1,T) | prow(R - 1,T) :- prow(R,T-1), pmove(0, T).
pcol(C,T) :- pcol(C,T-1), pmove(0, T).
pcol(C + 1,T) | pcol(C - 1,T) :- pcol(C,T-1), pmove(1, T).
prow(R,T) :- prow(R,T-1), pmove(1, T).

% Known walls forbid player moves and invalidate ghost counterexamples.
:- prow(R,T), pcol(C,T), wall(C,R).
sat :- grow(G,R,T), gcol(G,C,T), wall(C,R).

%%% P_check %%%

% movement of the ghosts
gmove(G,0, T) | gmove(G,1, T) :- G=0..ghosts-1, T=1..horizon - 1.
grow(G,R + 1,T) | grow(G,R - 1,T) :- grow(G,R,T-1), gmove(G,0, T).
gcol(G,C,T) :- gcol(G,C,T-1), gmove(G,0, T).
gcol(G,C + 1,T) | gcol(G,C - 1,T) :- gcol(G,C,T-1), gmove(G,1, T).
grow(G,R,T) :- grow(G,R,T-1), gmove(G,1, T).

% check saturation criteria
ok(0).
ok(0,G) :- G=0..ghosts-1.
sat :- ok(horizon-1).
:- not sat.

% saturation technique
gcol(G,C,T) :- C = -radius..radius, sat, T=1..horizon-1, G=0..ghosts-1.
grow(G,R,T) :- R = -radius..radius, sat, T=1..horizon-1, G=0..ghosts-1.
gmove(G,0,T) :- G=0..ghosts-1, T=1..horizon-1, sat.
gmove(G,1,T) :- G=0..ghosts-1, T=1..horizon-1, sat.

% each ghost either needs to be safely away from player or give penalty
% also consider ghosts previous position here, as pacman cannot "swap places"
% with ghosts like the agent can in gardener
ok(T,G,0) :- pcol(C,T), gcol(G,C',T-1), C != C', ok(T-1).
ok(T,G,1) :- pcol(C,T), gcol(G,C',T), C != C', ok(T-1).
ok(T,G,0) :- prow(R,T), grow(G,R',T-1), R != R', ok(T-1).
ok(T,G,1) :- prow(R,T), grow(G,R',T), R != R', ok(T-1).
ok(T,G) :- ok(T,G,0), ok(T,G,1).
ok(T,G) :- goutside(G), T=1..horizon-1.
good(T,G) | bad(T,G) :- G=0..ghosts-1, T=1..horizon-1.
ok(T,G) :- bad(T,G).
% Bound the recursion explicitly for NPC Gym layouts without ghosts.
ok(T) :- T=1..horizon-1, {ok(T,G) : G=0..ghosts-1} = ghosts, ok(T-1).

% penalty for violating the norm of eating a ghost
penalty(P, T) :- P = horizon - T, bad(T,F).


% Link the shared planner's zero-based actions to the reference's coordinates.
candidate(T,1..4) :- time(T).
:- action(T,1), prow(R,T), not prow(R+1,T+1).
:- action(T,2), prow(R,T), not prow(R-1,T+1).
:- action(T,3), pcol(C,T), not pcol(C+1,T+1).
:- action(T,4), pcol(C,T), not pcol(C-1,T+1).

% The reference charges once per time, even when several ghosts need a waiver.
cost("violations",P,T) :- penalty(P,T).

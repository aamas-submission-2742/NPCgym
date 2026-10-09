% Local action plans; walls/puddles block proposals, which execute as Stay.
candidate(T,0..4) :- time(T).
player(T+1,U,V) :- time(T), player(T,X,Y), action(T,A), move(X,Y,A,U,V,_).
effective(T,E) :- time(T), player(T,X,Y), action(T,A), move(X,Y,A,_,_,E).
:- player(T,X,Y), not window(X,Y).

% Puddles refill before drainage, including on the same input.
refilled(T,I) :- time(T), water(T,I,1,_).
refilled(T,I) :- time(T), water(T,I,_,1).
drain(T,I) :- refilled(T,I), puddle(I,X,Y), player(T+1,U,V), |X-U|+|Y-V|=1.
water(T+1,I,0,R) :- drain(T,I), puddle_respawn(R), time(T+1).
water(T+1,I,1,0) :- refilled(T,I), not drain(T,I), time(T+1).
water(T+1,I,0,R-1) :- time(T+1), water(T,I,0,R), R > 1, not drain(T,I).
water(T+1,I,0,0) :- time(T+1), water(T,I,0,0), not drain(T,I).

% Each uncollected frog follows every legal cardinal move, or stays if frozen.
% Collected branches disappear before drainage and cannot collect again.
moved(T,F,X,Y,K) :- time(T), frog(T,F,X,Y,K), K > 0.
moved(T,F,U,V,0) :- time(T), frog(T,F,X,Y,0), hop(X,Y,U,V).
collected(T,F) :- moved(T,F,X,Y,_), player(T+1,X,Y).
survives(T,F,X,Y,K) :- moved(T,F,X,Y,K), not player(T+1,X,Y).
near(T,I,X,Y) :- drain(T,I), puddle(I,U,V), survives(T,_,X,Y,_),
    |X-U| <= 1, |Y-V| <= 1, |X-U|+|Y-V| > 0.
frozen(T,X,Y) :- near(T,_,X,Y).
frog(T+1,F,X,Y,K-1) :- survives(T,F,X,Y,K), K > 0, not frozen(T,X,Y), time(T+1).
frog(T+1,F,X,Y,0) :- survives(T,F,X,Y,0), not frozen(T,X,Y), time(T+1).
frog(T+1,F,X,Y,R) :- survives(T,F,X,Y,_), frozen(T,X,Y), freeze(R), time(T+1).

% Alternative collection times belong to the SAME event: one cost per frog.
cost("violations",1,collect(F)) :- recipe(no_collect), collected(_,F).
cost("violations",1,collect(F)) :- recipe(permission_aware), collected(T,F), not effective(T,0).
% Distinct harmful drainages remain distinct events, even near the same frog.
cost("violations",1,drain(T,I)) :- recipe(drain), near(T,I,_,_).

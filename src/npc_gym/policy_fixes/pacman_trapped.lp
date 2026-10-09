% One-step Trapped abstraction: history is updated only from real observations.
#external active.
#external outside(A) : A=0..4.
candidate(0,0..4).
cost("violations",1,trapped) :- active, action(0,A), outside(A).

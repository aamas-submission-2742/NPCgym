% Domain interface: candidate(T,A), cost(Name,Value,Key).
% All optimization lives here; domain programs provide costs and constraints.
1 { action(T,A) : candidate(T,A), available_action(A) } 1 :- time(T).
:- action(0,A), not npc_allowed(A).
cost("policy",R,first_action) :- action(0,A), preference(A,R).
#minimize { V*W@P,N,K : cost(N,V,K), npc_objective(N,W,P) }.
% Earlier action IDs break ties before later ones, below all user priorities.
#minimize { A@-T,T : action(T,A) }.
#show action/2.
#show cost/3.

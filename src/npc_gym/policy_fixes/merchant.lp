% The current input activates a prohibition on the next Extract action.
#external at_tree.
#external has_wood.
candidate(0,0..6).
cost("violations",1,extract) :- action(0,4), at_tree, has_wood.

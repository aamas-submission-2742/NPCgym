#external warning_due.
candidate(0,0..6).
cost("violations",1,warn) :- warning_due, action(0,A), A != 6.

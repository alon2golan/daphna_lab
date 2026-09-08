# Professor PDF Notes - 2026-05-28

Extracted from `Research Project Summary with notes.pdf`.

## Page 13: Density normalization / degree definition
is the degree

## Page 13: Density normalization / degree definition
remind me how the degree is computed? is it the degree of the clipped RBF graph? did you try degree = sum_j K_ij?

## Page 13: Density normalization / alpha definition
additionally, is this the alpha you use in the results graphs below?

## Page 14: Density-normalization results
can you compute the average rank for each budget? The reason for this question is that the effect of normalization is expected to decrease as the ranks get close to 1.

## Page 14: Low-budget alpha benefit / delta and ProbCover comparison
1) in the budget range in which normalization is beneficial, 100-400, can you check sensitivity to delta? is the decrease in benefit larger or smaller when using non-optimal delta? 2) How does performance compares to ProbCover? preferably, just add the results of ProbCover to the plots

## Page 15: Symmetric alpha normalization formula
do I understand correctly that alpha is the actual power, and therefore alpha=0.5 is the geometric mean of the rank of the two edge-nodes?

## Page 16: Unclear phrase in document
meaning?

## Page 17: Experiment request
can you repeat this experiment with a single sigma (global K)?

## Page 17: Algorithm definition
is this the definition of greedy max coverage?

## Page 17: Wording of alpha result
"does not persist", or "degrades later performance"?

## Page 17: Degree vs local kernel interpretation
These scores are not totally different: degree and local kernel are very closely related, and both are affected by local density; right?

## Page 18: Important next graph
This is an important observation, with multiple ways to go about it - possibly, as the degree becomes uniformly very small (as low as 1-2), we should switch strategy. I would like to see, towards our next meeting, a graph of the average degree of the graph, and standard deviation, as a function of budget.

## Page 19: Degree-based switching / random mixture
the degree may also be used to decide when the methods stops being useful, and should be replaced by random selection (or a mixture strategy), a question targeted in Guy Hacohen's 2023 NeurIPS paper

## Page 19: 3-phase experiment clarity
isn't this what you did in the 3 phase experiments? as you can see, I didn't understand those experiments...

## Page 21: Delta schedule and ProbCover Fig 9
How does this heuristics corresponds to the empirical observation that delta should be decreased when the budget is increased (Fig 9 in the ProbCover paper)? One way to think about it is to assume that as delta is decreased, points - previously eliminated from the graph - return to the graph and therefore the average degree increases.

## Page 22: General feedback
Very thoughtful and well put.

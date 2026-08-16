function flag = dominates(a, b)
% DOMINATES  True if objective vector a Pareto-dominates b (minimization).

    flag = all(a <= b) && any(a < b);
end

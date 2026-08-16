function [X_pareto, F_pareto] = run_nsga2(objfun, lb, ub, pop_size, max_iter)
% RUN_NSGA2  Self-implemented NSGA-II (non-dominated sorting genetic
% algorithm II), using the same population size and iteration count as the
% MOEA/D and MOPSO implementations for a fair comparison.
%
%   [X_pareto, F_pareto] = run_nsga2(objfun, lb, ub, pop_size, max_iter)

    num_var = length(lb);
    num_obj = 3;

    crossover_prob = 0.9;          % SBX crossover probability
    mutation_prob  = 1 / num_var;  % polynomial mutation probability
    eta_c = 20;                    % SBX distribution index
    eta_m = 20;                    % mutation distribution index

    % Initialisation.
    X = rand(pop_size, num_var) .* (ub - lb) + lb;
    F = zeros(pop_size, num_obj);
    for i = 1:pop_size
        F(i, :) = objfun(X(i, :));
    end

    for iter = 1:max_iter
        [rank, crowding] = nsga2_sort(F);

        % Generate offspring via tournament selection + SBX + mutation.
        off_X = zeros(pop_size, num_var);
        for i = 1:2:pop_size
            p1 = tournament_select(rank, crowding);
            p2 = tournament_select(rank, crowding);
            if rand < crossover_prob
                [c1, c2] = sbx_crossover(X(p1, :), X(p2, :), lb, ub, eta_c);
            else
                c1 = X(p1, :); c2 = X(p2, :);
            end
            c1 = polynomial_mutation(c1, lb, ub, eta_m, mutation_prob);
            c2 = polynomial_mutation(c2, lb, ub, eta_m, mutation_prob);
            off_X(i, :) = c1;
            if i + 1 <= pop_size
                off_X(i + 1, :) = c2;
            end
        end

        % Evaluate offspring.
        off_F = zeros(pop_size, num_obj);
        for i = 1:pop_size
            off_F(i, :) = objfun(off_X(i, :));
        end

        % Combine parents and offspring, keep the best pop_size.
        X_comb = [X; off_X];
        F_comb = [F; off_F];
        [rank2, crowding2] = nsga2_sort(F_comb);
        [~, order] = sortrows([rank2, -crowding2]);
        X = X_comb(order(1:pop_size), :);
        F = F_comb(order(1:pop_size), :);
    end

    % Return the non-dominated front of the final population.
    nd = get_nondominated_flag(F);
    X_pareto = X(nd, :);
    F_pareto = F(nd, :);
end

function [rank, crowding] = nsga2_sort(F)
% Fast non-dominated sort + crowding distance.
    n = size(F, 1);
    rank = inf(n, 1);
    dominated_by = zeros(n, 1);
    dominates_list = cell(n, 1);
    for i = 1:n
        for j = 1:n
            if i ~= j
                if dominates(F(j, :), F(i, :))
                    dominated_by(i) = dominated_by(i) + 1;
                elseif dominates(F(i, :), F(j, :))
                    dominates_list{i} = [dominates_list{i}, j];
                end
            end
        end
    end

    front = 1;
    current = find(dominated_by == 0);
    while ~isempty(current)
        rank(current) = front;
        next_front = [];
        for idx = current(:)'
            for q = dominates_list{idx}
                dominated_by(q) = dominated_by(q) - 1;
                if dominated_by(q) == 0
                    next_front = [next_front, q];
                end
            end
        end
        current = unique(next_front);
        front = front + 1;
    end

    % Crowding distance per front.
    crowding = zeros(n, 1);
    for f = 1:(front - 1)
        members = find(rank == f);
        m = numel(members);
        if m <= 2
            crowding(members) = inf;
            continue;
        end
        for obj = 1:size(F, 2)
            [~, order] = sort(F(members, obj));
            sm = members(order);
            crowding(sm(1)) = inf;
            crowding(sm(end)) = inf;
            fmin = F(sm(1), obj);
            fmax = F(sm(end), obj);
            if fmax > fmin
                for k = 2:(m - 1)
                    crowding(sm(k)) = crowding(sm(k)) + ...
                        (F(sm(k + 1), obj) - F(sm(k - 1), obj)) / (fmax - fmin);
                end
            end
        end
    end
end

function idx = tournament_select(rank, crowding)
% Binary tournament selection (lower rank, then higher crowding wins).
    n = numel(rank);
    a = randi(n);
    b = randi(n);
    if rank(a) < rank(b) || (rank(a) == rank(b) && crowding(a) > crowding(b))
        idx = a;
    else
        idx = b;
    end
end

function [c1, c2] = sbx_crossover(x1, x2, lb, ub, eta_c)
% Simulated Binary Crossover (SBX).
    num_var = numel(x1);
    c1 = x1; c2 = x2;
    u = rand(1, num_var);
    for j = 1:num_var
        if u(j) <= 0.5
            beta = (2 * u(j))^(1 / (eta_c + 1));
        else
            beta = (1 / (2 * (1 - u(j))))^(1 / (eta_c + 1));
        end
        c1(j) = 0.5 * ((1 + beta) * x1(j) + (1 - beta) * x2(j));
        c2(j) = 0.5 * ((1 - beta) * x1(j) + (1 + beta) * x2(j));
        c1(j) = min(max(c1(j), lb(j)), ub(j));
        c2(j) = min(max(c2(j), lb(j)), ub(j));
    end
end

function x = polynomial_mutation(x, lb, ub, eta_m, mutation_prob)
% Polynomial mutation.
    num_var = numel(x);
    for j = 1:num_var
        if rand < mutation_prob
            u = rand;
            if u < 0.5
                delta = (2 * u)^(1 / (eta_m + 1)) - 1;
            else
                delta = 1 - (2 * (1 - u))^(1 / (eta_m + 1));
            end
            x(j) = x(j) + delta * (ub(j) - lb(j));
            x(j) = min(max(x(j), lb(j)), ub(j));
        end
    end
end

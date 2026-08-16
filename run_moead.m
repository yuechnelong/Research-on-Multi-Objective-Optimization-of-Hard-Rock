function [X_pareto, F_pareto] = run_moead(objfun, lb, ub, pop_size, max_iter, neighbor_num)
% RUN_MOEAD  Multi-objective evolutionary algorithm based on decomposition.

    num_var = length(lb);
    num_obj = 3;

    W = rand(pop_size, num_obj);
    W = W ./ sum(W, 2);

    distW = squareform(pdist(W));
    [~, B] = sort(distW, 2);
    B = B(:, 1:neighbor_num);

    X = rand(pop_size, num_var) .* (ub - lb) + lb;
    F = zeros(pop_size, num_obj);
    for i = 1:pop_size
        F(i, :) = objfun(X(i, :));
    end

    z = min(F, [], 1);

    for iter = 1:max_iter
        for i = 1:pop_size
            P = B(i, randperm(neighbor_num, 2));
            x1 = X(P(1), :);
            x2 = X(P(2), :);

            y = x1 + rand(1, num_var) .* (x2 - x1);
            y = min(max(y, lb), ub);

            mutation_rate = 1 / num_var;
            for j = 1:num_var
                if rand < mutation_rate
                    y(j) = lb(j) + rand * (ub(j) - lb(j));
                end
            end

            fy = objfun(y);
            z = min(z, fy);

            for jj = 1:neighbor_num
                k = B(i, jj);
                g_old = max(W(k, :) .* abs(F(k, :) - z));
                g_new = max(W(k, :) .* abs(fy - z));
                if g_new <= g_old
                    X(k, :) = y;
                    F(k, :) = fy;
                end
            end
        end
    end

    nd = get_nondominated_flag(F);
    X_pareto = X(nd, :);
    F_pareto = F(nd, :);
end

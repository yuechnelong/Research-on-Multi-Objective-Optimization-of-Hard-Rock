function [X_pareto, F_pareto] = run_mopso(objfun, lb, ub, pop_size, max_iter, archive_size)
% RUN_MOPSO  Multi-objective particle swarm optimisation with an external
% archive of non-dominated solutions.

    num_var = length(lb);
    num_obj = 3;

    X = rand(pop_size, num_var) .* (ub - lb) + lb;
    V = zeros(pop_size, num_var);

    F = zeros(pop_size, num_obj);
    for i = 1:pop_size
        F(i, :) = objfun(X(i, :));
    end

    pbest_X = X;
    pbest_F = F;

    archive_X = X;
    archive_F = F;
    nd = get_nondominated_flag(archive_F);
    archive_X = archive_X(nd, :);
    archive_F = archive_F(nd, :);

    w = 0.6; c1 = 1.5; c2 = 1.5;

    for iter = 1:max_iter
        for i = 1:pop_size
            leader_id = randi(size(archive_X, 1));
            leader = archive_X(leader_id, :);

            V(i, :) = w * V(i, :) ...
                + c1 * rand(1, num_var) .* (pbest_X(i, :) - X(i, :)) ...
                + c2 * rand(1, num_var) .* (leader - X(i, :));

            X(i, :) = X(i, :) + V(i, :);
            X(i, :) = min(max(X(i, :), lb), ub);

            F(i, :) = objfun(X(i, :));

            if dominates(F(i, :), pbest_F(i, :))
                pbest_X(i, :) = X(i, :);
                pbest_F(i, :) = F(i, :);
            elseif ~dominates(pbest_F(i, :), F(i, :)) && rand < 0.5
                pbest_X(i, :) = X(i, :);
                pbest_F(i, :) = F(i, :);
            end
        end

        archive_X = [archive_X; X];
        archive_F = [archive_F; F];
        nd = get_nondominated_flag(archive_F);
        archive_X = archive_X(nd, :);
        archive_F = archive_F(nd, :);

        if size(archive_X, 1) > archive_size
            select_id = randperm(size(archive_X, 1), archive_size);
            archive_X = archive_X(select_id, :);
            archive_F = archive_F(select_id, :);
        end
    end

    X_pareto = archive_X;
    F_pareto = archive_F;
end

function nd_flag = get_nondominated_flag(F)
% GET_NONDOMINATED_FLAG  Logical flag of non-dominated rows of objective
% matrix F (n x n_obj, minimization).

    n = size(F, 1);
    nd_flag = true(n, 1);
    for i = 1:n
        for j = 1:n
            if i ~= j && dominates(F(j, :), F(i, :))
                nd_flag(i) = false;
                break;
            end
        end
    end
end

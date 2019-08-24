function [rank, score] = ECFS_Zeal(train_X, train_y, alpha)

    sample_negative = train_X(train_y == -1, :);
    sample_positive = train_X(train_y == 1, :);
    mu_sample_negative = mean(sample_negative);
    mu_sample_positive = mean(sample_positive);
    
    
    %% Metric 1: Mutual Information
    mi_s = [];
    for i = 1:size(train_X, 2)
        mi_s = [mi_s, Mutual_Information(train_X(:, i), train_y)];
    end
    
    %% Metric 2: Class Separation
    sep_scores = ([mu_sample_positive - mu_sample_negative].^2);
    std_positive = std(sample_positive).^2;
    total_std = std_positive + std(sample_negative).^2;
    f = find(total_std == 0);  %% remove ones where nothing occurs
    total_std(f) = 10000;
    sep_scores = sep_scores ./ total_std;
    
    %% Building the graph
    vec =  abs(sep_scores + mi_s) / 2;
    
    %% Building the graph
    Kernel_ij = [vec' * vec];
    
    Kernel_ij = Kernel_ij - min(min(Kernel_ij));
    Kernel_ij = Kernel_ij ./ max(max(Kernel_ij));
    
    %% Standard Deviation
    STD = std(train_X, [], 1);
    STDMatrix = bsxfun(@max, STD, STD');
    STDMatrix = STDMatrix - min(min(STDMatrix));
    sigma_ij = STDMatrix ./ max(max(STDMatrix));
    
    Kernel = (alpha * Kernel_ij + (1 - alpha) * sigma_ij);
    
    
    %% Eigenvector Centrality and Ranking
    [eigVect, ~] = eigs(double(Kernel), 1, 'lm');
    [~, rank] = sort(abs(eigVect), 'descend');
    score = eigVect;
    
end
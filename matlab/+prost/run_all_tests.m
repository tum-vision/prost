function run_all_tests()

    prost.init();
    
    unit_tests = { 
        'linop_diags'; ...
        'linop_gradient2d'; ...
        'linop_gradient3d'; ...
        'linop_sparse_zero'; ...
        'prox_sum_ind_simplex'; ...
        'prox_sum_norm2'; ...
        'prox_transform'; ...
        'prox_sum_ind_epi_polyhedral' };

    num_passed = 0;
    for i=1:size(unit_tests,1)
        fprintf('Running unit test "%s"...', unit_tests{i});
        passed = eval(['prost.test.test_', unit_tests{i}]);
        
        if passed
            fprintf(' success!\n');
            num_passed = num_passed + 1;
        end
    end

    fprintf('%1d/%1d unit tests passed.\n', ...
            num_passed, size(unit_tests, 1));
    
    prost.release();

end

unit_tests = { 'linop_sparse'; 'prox_simplex'; 'prox_norm2' };

num_passed = 0;
for i=1:size(unit_tests,1)
    fprintf('Running unit test "%s"...', unit_tests{i});
    passed = eval(['test.test_', unit_tests{i}]);
    
    if passed
        fprintf(' success!\n');
        num_passed = num_passed + 1;
    end
end

fprintf('%1d/%1d unit tests passed.\n', num_passed, size(unit_tests, 1));
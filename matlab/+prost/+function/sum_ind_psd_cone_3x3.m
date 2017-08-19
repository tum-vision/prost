function [func] = sum_ind_psd_cone_3x3(interleaved)
% SUM_IND_PSD_CONE_3x3  func = sum_ind_psd_cone_3x3(interleaved)
%

    dim = 9;
    func = @(idx, count) { 'elem_operation:ind_psd_cone_3x3', idx, count, false, { count / dim, dim, interleaved } };
end

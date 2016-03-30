function set_dual_pair(var_p, var_d, linop)
    var_p.pairing{end+1} = var_d;
    var_p.linop{end+1} = linop;
end

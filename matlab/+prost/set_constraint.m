function set_constraint(var, var_constr, linop)
    var.pairing{end+1} = var_constr;
    var.linop{end+1} = linop;
end

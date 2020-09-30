function [ str_varName ] = getVarName( var )

    str_varName = sprintf('%s', inputname(1));

end
function l = getLikelihood(f, a, b, type)
if type==0
    max_f = max(f);
    d = ((max_f+b)./f) - 1;
    l = exp(-a.*d.*d);
elseif type==1
    max_f = max(f);
    d = 1 - (f./(max_f + b));
    l = exp(-a.*d.*d);
else
    l = exp(a.*f);
end
end



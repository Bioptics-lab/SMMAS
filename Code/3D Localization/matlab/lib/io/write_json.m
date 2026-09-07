function write_json(path, s)
%WRITE_JSON Write a MATLAB struct as UTF-8 JSON.
txt = jsonencode(s);
fid = fopen(path, 'w');
if fid < 0
    error('cannot write %s', path);
end
fwrite(fid, txt, 'char');
fclose(fid);
end

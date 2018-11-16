import bencoder
import bencode
import codecs
out_ip = '222.333.444.55'
file_path = "C:\Users\Tommy\AppData\Roaming\uTorrent\settings.dat"
out_file_path = "C:\Users\Tommy\AppData\Roaming\uTorrent\settings2.dat"
f = codecs.open(file_path, "rb").read()
_id1 = f.find('net.bind_ip')
print _id1
_id2 = f[_id1:].find(':')
print _id2

print f[_id1:_id1+5]
# d = bencoder.decode(f.read())
d = bencode.bdecode(f)
# del d[b"info"][b"pieces"] # That's a long hash
from pprint import pprint
pprint(d)

d['net.bind_ip'] = out_ip
d['net.outgoing_ip'] = out_ip


f_out = bencode.bencode(d)
codecs.open(out_file_path, "wb").write(f_out)





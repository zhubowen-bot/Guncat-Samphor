import re, os, glob, base64, struct
p = glob.glob("*.html")[0]
s = open(p, encoding="utf-8", errors="replace").read()
i = s.index("const weights_b64")
j = s.index("const config", i)
blk = s[i:j]
pat = re.compile(r'"([\w\.]+)"\s*:\s*\{\s*"size"\s*:\s*\[([^\]]*)\]\s*,\s*"data"\s*:\s*"([^"]*)"', re.S)
total = 0
for m in pat.finditer(blk):
    name=m.group(1); size=[int(x) for x in m.group(2).replace("\n"," ").split(',') if x.strip()]; ln=len(m.group(3))
    n=1
    for x in size: n*=x
    total += n
    print(name, size, "elems", n, "b64", ln)
print("total params", total)
print("expected bytes", total*4, "actual b64 chars", len(blk))
m = re.search(r"const vocab = \[(.*?)\]", s, re.S)
v = m.group(1)
items = re.findall(r'"(?:[^"\\]|\\.)*"', v)
print("vocab items:", len(items))
print("file size MB", os.path.getsize(p)/1e6)

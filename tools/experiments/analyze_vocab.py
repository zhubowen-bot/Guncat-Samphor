import re, glob, collections
p = glob.glob("*.html")[0]
s = open(p, encoding="utf-8", errors="replace").read()
m = re.search(r"const vocab = \[(.*?)\]", s, re.S)
v = m.group(1)
items = re.findall(r'"(?:[^"\\]|\\.)*"', v)
chars = [eval(x) for x in items]
print("vocab len", len(chars))
print("unique", len(set(chars)))
print("dups", [c for c,n in collections.Counter(chars).items() if n>1])
print("first 40", chars[:40])
print("last 40", chars[-40:])
print("non-BMP", [c for c in chars if len(c)!=1])
print("max ord", max(ord(c) for c in chars), "min", min(ord(c) for c in chars))
# count CJK
print("cjk", sum(1 for c in chars if '\u4e00' <= c <= '\u9fff'))
print("latin", sum(1 for c in chars if c.isascii() and c.isalpha()))
print("digits", sum(1 for c in chars if c.isdigit()))

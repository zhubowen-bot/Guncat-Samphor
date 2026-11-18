import { AutoTokenizer, AutoModelForCausalLM } from '@huggingface/transformers';
const id = 'onnx-community/Qwen2.5-0.5B';
const tok = await AutoTokenizer.from_pretrained(id);
const model = await AutoModelForCausalLM.from_pretrained(id, { dtype: 'q4', device: 'cpu' });
const prompt = '傍晚的时候，我走在回家的路上。';
let total = 0, match = 0, prefixMatch = 0;
for (let k = 0; k < 20; k++) {
  const inputs = tok(prompt, { add_special_tokens: false });
  const out = await model.generate({ ...inputs, max_new_tokens: 50, do_sample: true, temperature: 0.8, top_p: 0.95, repetition_penalty: 1.1 });
  const ids = out.tolist()[0].map(Number).slice(inputs.input_ids.dims[1]);
  const text = tok.decode(ids, { skip_special_tokens: true });
  const re = tok.encode(text, { add_special_tokens: false });
  const m = ids.length === re.length && ids.every((v, i) => v === re[i]);
  total++;
  if (m) match++;
  // prefix match length
  let p = 0; while (p < ids.length && p < re.length && ids[p] === re[p]) p++;
  prefixMatch += p / ids.length;
  console.log(k, 'len', ids.length, 're', re.length, 'exact', m, 'prefix', (p/ids.length).toFixed(2));
  console.log('  text:', text.slice(0, 40));
}
console.log('exact match', match, '/', total, 'avg prefix', (prefixMatch/total).toFixed(3));

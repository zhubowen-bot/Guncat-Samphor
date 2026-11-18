import { AutoTokenizer, AutoModelForCausalLM } from '@huggingface/transformers';
const id = 'onnx-community/Qwen2.5-0.5B';
const tok = await AutoTokenizer.from_pretrained(id);
const model = await AutoModelForCausalLM.from_pretrained(id, { dtype: 'q4', device: 'cpu' });
const preamble = '傍晚的时候，我走在回家的路上。';
const hiddens = ['', '以下是一段自然流畅的中文：', '这是一段关于日常生活的短文：', '用叙述的方式写下去：'];
for (const h of hiddens) {
  const inputs = tok(h + preamble, { add_special_tokens: false });
  const out = await model.generate({ ...inputs, max_new_tokens: 60, do_sample: true, temperature: 0.9, top_p: 0.98, repetition_penalty: 1.05 });
  const ids = out.tolist()[0].map(Number).slice(inputs.input_ids.dims[1]);
  console.log('\n--- hidden: ' + (h || '(none)') + ' ---\n' + preamble + tok.decode(ids, { skip_special_tokens: true }));
}

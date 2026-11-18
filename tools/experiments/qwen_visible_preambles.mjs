import { AutoTokenizer, AutoModelForCausalLM } from '@huggingface/transformers';
const id = 'onnx-community/Qwen2.5-0.5B';
const tok = await AutoTokenizer.from_pretrained(id);
const model = await AutoModelForCausalLM.from_pretrained(id, { dtype: 'q4', device: 'cpu' });
const hidden = '以下是一段自然流畅的中文：';
const preambles = [
  '天气很好，',
  '傍晚的时候，我走在回家的路上。',
  '清晨的阳光透过窗帘照进房间。',
  '生活中有很多值得记录的小事。',
  '最近发生了一些小事。',
  '多年以后回想起来，',
];
for (const preamble of preambles) {
  const inputs = tok(hidden + preamble, { add_special_tokens: false });
  const out = await model.generate({ ...inputs, max_new_tokens: 60, do_sample: true, temperature: 0.9, top_p: 0.98, repetition_penalty: 1.05 });
  const ids = out.tolist()[0].map(Number).slice(inputs.input_ids.dims[1]);
  console.log('\n--- ' + preamble + ' ---\n' + preamble + tok.decode(ids, { skip_special_tokens: true }));
}

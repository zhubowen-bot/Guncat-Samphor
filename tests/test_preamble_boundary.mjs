import { encode2m, decode2m } from '../src/stego2m.js';

const cases = [
  '刚下飞机，人在火星，',
  '谢邀，',
  '天气很好',
  '家人们谁懂啊，',
  '专家提醒：',
  '这是一个没有标点符号的二十四字开头示例文本内容',
  '这，是，一个，有很多逗号的开头，',
];
for (const pre of cases) {
  console.log('\n=== preamble', JSON.stringify(pre), 'len', Array.from(pre).length, '===');
  try {
    const enc = await encode2m('测试边界识别。', { preamble: pre, profileId: 0, attempts: 1 });
    console.log('used preamble:', JSON.stringify(enc.stats.preamble), 'body', enc.stats.bodyChars);
    console.log(enc.text.slice(0, 80));
    const dec = await decode2m(enc.text, { profileId: 0 });
    console.log('decoded', dec.text === '测试边界识别。', 'preambleEnd', dec.stats.preambleEnd, 'ok');
  } catch (e) { console.log('ERROR', e.message); }
}

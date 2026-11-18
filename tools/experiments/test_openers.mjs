import { encode2m } from '../src/stego2m.js';

const openers = [
  '天气很好，',
  '生活中有很多值得记录的小事。',
  '傍晚的时候，我走在回家的路上。',
  '最近发生了一些小事。',
  '在我看来，',
  '人工智能正在改变我们的生活。',
  '以下是我的一些想法，',
  '今天想和你聊聊生活，',
  '多年以后回想起来，',
  '专家提醒：',
  '最新数据显示，',
];
for (const p of openers) {
  const r = await encode2m('明天下午三点，老地方见。', { preamble: p, profileId: 0, attempts: 3 });
  console.log('\n===', p, '===');
  console.log(r.text);
  console.log('body', r.stats.bodyChars, 'bits/char', r.stats.bitsPerChar.toFixed(2), 'score', r.stats.qualityScore.toFixed(3), 'ms', r.stats.encodeMs);
}

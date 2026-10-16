export const meta = {
  name: 'iterate-writeup',
  description: 'One round: N candidate writeups → 4 rubric judges each → pick, revise, re-judge, save state',
  whenToUse: 'Run one improvement round of the Big Data Bowl writeup. args: {round, candidates, angles, threshold}',
  phases: [
    { title: 'Prepare', detail: 'data present, read previous state' },
    { title: 'Author', detail: 'one candidate per angle', model: 'opus' },
    { title: 'Judge', detail: '4 rubric judges per candidate' },
    { title: 'Revise', detail: 'merge best + feedback, re-judge' },
    { title: 'Record', detail: 'state/round_N.json, writeup/writeup.md' },
  ],
}

const round = (args && args.round) || 1
const nCand = (args && args.candidates) || 3
const threshold = (args && args.threshold) || 8.5
const DEFAULT_ANGLES = [
  'DL/EDGE: pass-rush drill and run-the-hoop kinematics (get-off burst, bend) vs in-game get-off and pressure rate; year-standardized; baseline = 10-yd split + 3-cone + NGS + pick.',
  'WR/TE: break mechanics from route drills (decel into break, speed through break, re-acceleration) matched to the same route family in games (OUT/HITCH/IN/SLANT/CORNER), vs separation over expected; one named metric.',
  'Movement translation: for each player, similarity between drill speed/accel profile and his in-game profile on the same movement type (using game_frames_sample and combine_frames); does translation explain production beyond pick? Pick the group with the clearest signal.',
  'Value over draft slot: expected opportunity and production from pick + Combine numbers per group; which position group the market mis-prices most; named over/under-performers; what tracking adds on top.',
]
const angles = (args && args.angles) || DEFAULT_ANGLES.slice(0, nCand)
const dir = `writeup/candidates/round_${String(round).padStart(2, '0')}`

const JUDGE = { type: 'object', properties: {
  score: { type: 'number' }, subscores: { type: 'object' }, justification: { type: 'string' },
  top_changes: { type: 'array', items: { type: 'string' } }, fatal_flaws: { type: 'array', items: { type: 'string' } },
}, required: ['score', 'top_changes', 'fatal_flaws'] }
const AUTHOR = { type: 'object', properties: { path: { type: 'string' }, summary: { type: 'string' }, ok: { type: 'boolean' } }, required: ['path', 'summary', 'ok'] }
const WEIGHTS = { 'judge-football': 0.3, 'judge-datascience': 0.3, 'judge-writeup': 0.2, 'judge-viz': 0.2 }

function weighted(verdicts) {
  let s = 0, w = 0
  for (const [name, v] of Object.entries(verdicts)) { if (v) { s += WEIGHTS[name] * v.score; w += WEIGHTS[name] } }
  return w ? +(s / w).toFixed(2) : 0
}

phase('Prepare')
const prep = await agent(
  `Make sure the data is present: if data/processed/combine_events.parquet is missing, run \`bash cloud/setup.sh\`. Then read state/latest.json (if it exists) and writeup/writeup.md (if it exists) and return a compact brief for authors: previous best score per rubric component, the judges' top_changes, what the current writeup claims, and its path. Also return the list of candidate directories from previous rounds. Keep it under 400 words.`,
  { phase: 'Prepare', effort: 'low' })

phase('Author')
const judgeAll = (cand, i) => parallel(Object.keys(WEIGHTS).map((j) => () =>
  agent(`Judge the candidate writeup at ${cand.path}/writeup.md (figures in ${cand.path}/figures, analysis in ${cand.path}/analysis.py, numbers in ${cand.path}/numbers.json). Round ${round}, candidate ${i + 1}: ${angles[i] || 'revision'}. Previous-round brief:\n${prep}`,
    { agentType: j, label: `${j}:c${i + 1}`, phase: 'Judge', schema: JUDGE })))
  .then((vs) => { const verdicts = {}; Object.keys(WEIGHTS).forEach((j, k) => { verdicts[j] = vs[k] }); return { ...cand, verdicts, score: weighted(verdicts) } })

const judged = (await pipeline(
  angles.map((angle, i) => ({ angle, i })),
  ({ angle, i }) => agent(
    `Round ${round}, candidate ${i + 1} of ${angles.length}. Angle: ${angle}\n\nWrite the complete candidate into ${dir}/cand_${i + 1}/ (create it). Brief from the previous round:\n${prep}\n\nIf a previous writeup exists and this angle matches it, revise it instead of starting over, applying the judges' top_changes. Return ok=false if you could not produce a passing writeup/check.py result.`,
    { agentType: 'writeup-author', label: `author:c${i + 1}`, phase: 'Author', schema: AUTHOR }),
  (cand, item) => (cand && cand.ok ? judgeAll(cand, item.i) : null),
)).filter(Boolean)

if (!judged.length) { log('No candidate passed the checker'); return { round, ok: false } }
judged.sort((a, b) => b.score - a.score)
log(`Round ${round}: ` + judged.map((c, k) => `c${k + 1}=${c.score}`).join(', '))
const best = judged[0]

phase('Revise')
const feedback = Object.entries(best.verdicts).map(([j, v]) => `${j} (${v && v.score}): ${(v && v.top_changes || []).join(' | ')}${v && v.fatal_flaws && v.fatal_flaws.length ? ' FATAL: ' + v.fatal_flaws.join(' | ') : ''}`).join('\n')
const others = judged.slice(1).map((c) => `${c.path} (score ${c.score}): ${c.summary}`).join('\n')
const revised = await agent(
  `Revise the best candidate at ${best.path} into ${dir}/revised/ (copy everything, then improve). Apply the judges' feedback below in priority order, fix every fatal flaw, and graft the single strongest idea from the runners-up if it fits in the word budget. Keep one metric, one story.\n\nJudges:\n${feedback}\n\nRunners-up:\n${others || 'none'}\n\nReturn ok=false if writeup/check.py does not pass.`,
  { agentType: 'writeup-author', label: 'reviser', phase: 'Revise', schema: AUTHOR })
const final = revised && revised.ok ? await judgeAll(revised, -1) : best
const winner = final.score >= best.score ? final : best

phase('Record')
const record = await agent(
  `Record the round. Write state/round_${String(round).padStart(2, '0')}.json with: round, candidates (path, angle, score, per-judge scores, fatal_flaws), winner path and score, the full judge feedback for the winner, and a 5-line changelog. Write the same object to state/latest.json plus the previous rounds' scores as "history". If the winner's score (${winner.score}) is higher than the score in the previous state/latest.json (or there was none), copy ${winner.path}/writeup.md to writeup/writeup.md, its figures to writeup/figures/, its numbers.json to writeup/numbers.json and its analysis.py to analysis/writeup_analysis.py. Then \`git add -A && git commit -m "Writeup round ${round}: best ${winner.score}"\` (no Co-Authored-By). Do not push. Data:\n${JSON.stringify({ round, winner: { path: winner.path, score: winner.score, verdicts: winner.verdicts }, candidates: judged.map((c) => ({ path: c.path, score: c.score, verdicts: c.verdicts })) })}`,
  { phase: 'Record', effort: 'low' })

return { round, best: winner.score, winner: winner.path, done: winner.score >= threshold, candidates: judged.map((c) => ({ path: c.path, score: c.score })), record }

import { Link } from 'react-router-dom'
import { allLabs } from '../data/modules/index.js'
import { domainById } from '../data/exam.js'
import { useProgress } from '../lib/progress.js'
import { slugify } from '../components/Blocks.jsx'
import RichText from '../components/RichText.jsx'

export default function Labs() {
  const progress = useProgress()

  return (
    <div className="wrap">
      <span className="eyebrow">Hands-on labs</span>
      <h1 style={{ maxWidth: '22ch' }}>Learn by training real models.</h1>
      <p className="hero-sub" style={{ marginBottom: 26 }}>
        {allLabs.length} labs take you from a hand-written training loop to a GPT you built yourself, a pretrained small LLM,
        LoRA/QLoRA and DPO fine-tunes, benchmarks, quantization and an OpenAI-compatible server for your own model.
        Each one opens inside its lesson with the full code. Most run on a free Colab T4 or a laptop; the rest need one 24 GB GPU.
      </p>

      <div className="lab-list">
        {allLabs.map((lab, i) => {
          const d = domainById[lab.moduleId]
          const done = !!progress.lessons[`${lab.moduleId}/${lab.lessonId}`]
          return (
            <Link key={`${lab.lessonId}-${i}`} to={`/module/${lab.moduleId}/${lab.lessonId}`} className="card card-hover"
              onClick={() => setTimeout(() => document.getElementById(`lab-${slugify(lab.title)}`)?.scrollIntoView({ block: 'start' }), 60)}>
              <div className="mod-head">
                <div className="mod-idx" style={done ? { background: d.color, color: '#041018', borderColor: d.color } : { color: d.color, borderColor: d.color }}>
                  {done ? '✓' : String(i + 1).padStart(2, '0')}
                </div>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="mod-title">{lab.title}</div>
                  <div className="tiny muted" style={{ lineHeight: 1.55 }}><RichText>{lab.x}</RichText></div>
                  <div className="mod-meta">
                    <span className="pill" style={{ color: d.color, borderColor: d.color }}>Module {d.n} · {d.name}</span>
                    <span className="pill">{lab.time}</span>
                    <span className="pill">{lab.gpu}</span>
                  </div>
                </div>
              </div>
            </Link>
          )
        })}
      </div>

      <section style={{ marginTop: 40 }}>
        <h2 style={{ fontSize: '1.15rem' }}>Setting up</h2>
        <div className="card">
          <p className="muted tiny" style={{ lineHeight: 1.65 }}>
            <b>Fastest start:</b> open a new notebook at <a className="ext" href="https://colab.research.google.com/" target="_blank" rel="noreferrer">Google Colab</a>,
            choose <i>Runtime → Change runtime type → T4 GPU</i>, and paste each code block into a cell.
            <b> Locally:</b> Python 3.10+, then the packages below. Apple Silicon works for Module 1–2 labs via the <code className="inline-code">mps</code> device.
          </p>
          <div className="code" style={{ margin: 0 }}>
            <div className="code-h"><span>bash</span></div>
            <pre><code>{`python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets tokenizers accelerate peft trl bitsandbytes lm-eval
# serving (Linux + NVIDIA GPU)
pip install vllm`}</code></pre>
          </div>
        </div>
      </section>
    </div>
  )
}

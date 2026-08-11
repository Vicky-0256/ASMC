import React from "react";
import {
  Timer,
  Activity,
  BarChart3,
  Workflow,
  BookOpen,
  Table2,
  ShieldCheck,
  CheckCircle2,
  ExternalLink,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import HeroWithModelAnimation from "./ASMCMethodAnimation";

const NAV_ITEMS = [
  ["Abstract", "abstract"],
  ["Motivation", "motivation"],
  ["Method", "demo"],
  ["Results", "results"],
  ["System", "system"],
  ["Diagnostics", "collapse"],
];

const STATS = [
  ["MATH500 accuracy", "80.6%"],
  ["ASMC-adapt p95", "73.7s"],
  ["MCMC p95", "1318s"],
  ["Reorder speedup", "53–77×"],
];

const RESULT_ROWS = [
  { budget: "16×", method: "Best-of-4", acc: "79.2", p50: "14.5", p95: "60.3" },
  { budget: "16×", method: "ASMC N=16", acc: "77.6", p50: "14.2", p95: "38.5" },
  { budget: "128×", method: "MCMC", acc: "73.8", p50: "211.6", p95: "1317.5", badge: "tail risk" },
  { budget: "128×", method: "ASMC-adapt", acc: "80.6", p50: "20.9", p95: "73.7", badge: "best frontier" },
];

const COLLAPSE_ROWS = [
  { diagnostic: "ESSmin / N", interpretation: "Low values indicate particle degeneracy and predict failures." },
  { diagnostic: "Resampling events", interpretation: "Frequent resampling suggests diversity is being lost." },
  { diagnostic: "Resampling scheme", interpretation: "Residual vs. multinomial is secondary to collapse severity." },
];

const SYSTEM_FACTS = [
  { label: "Latency", value: "53–77×", detail: "KV reorder speedup over prefix replay" },
  { label: "Boundary regime", value: "N=96", detail: "reorder remains feasible at L=3072, R=50" },
  { label: "Rebuild failure", value: "N≥80", detail: "prefix replay hits OOM in the stress test" },
];

const COLLAPSE_FACTS = [
  { label: "Low diversity", value: "p=9.1e-16", detail: "ESSmin/N separates correct and incorrect runs" },
  { label: "High collapse", value: "43.9%", detail: "accuracy when resampling events reach ≥16" },
  { label: "Scheme effect", value: "+0.5%", detail: "residual vs multinomial is secondary" },
];

const TAKEAWAYS = [
  "Parallel particles replace a serial MCMC chain.",
  "ESS decides whether resampling is needed.",
  "Ancestor maps reorder KV caches instead of replaying prefixes.",
  "Adaptive-N spends more compute only on hard problems.",
];

const figureSrc = (fileName) => `${import.meta.env.BASE_URL}figures/${fileName}`;

function focusClass() {
  return "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 focus-visible:ring-offset-2 focus-visible:ring-offset-[#fcfbf8]";
}

function DesignSystemStyles() {
  return (
    <style>{`
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');
      .paper-page { --ink: #111827; --muted: #4b5563; --line: rgba(17, 24, 39, 0.14); --paper: rgba(255,255,255,0.96); --blue: #1f77b4; --green: #2ca02c; --red: #c43c39; font-family: 'Source Sans 3', ui-sans-serif, system-ui; }
      .paper-serif { font-family: 'Source Serif 4', Georgia, serif; }
      .paper-mono { font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace; }
      .paper-pretty { text-wrap: pretty; }
      .paper-card { border: 1px solid var(--line); background: var(--paper); box-shadow: 0 1px 2px rgba(17, 24, 39, 0.05); }
      .paper-rule { border-top: 1px solid rgba(17, 24, 39, 0.16); }
      .paper-grid { background-image: linear-gradient(to right, rgba(17,24,39,.035) 1px, transparent 1px), linear-gradient(to bottom, rgba(17,24,39,.035) 1px, transparent 1px); background-size: 32px 32px; }
      .figure-scroll { scrollbar-color: rgba(17,24,39,.28) transparent; scrollbar-width: thin; }
      .figure-scroll::-webkit-scrollbar { height: 8px; }
      .figure-scroll::-webkit-scrollbar-thumb { background: rgba(17,24,39,.24); border-radius: 999px; }
      .figure-scroll::-webkit-scrollbar-track { background: transparent; }
      @media (max-width: 640px) {
        .asmc-mobile-fit { width: min(calc(100vw - 72px), 318px) !important; max-width: min(calc(100vw - 72px), 318px) !important; }
        .asmc-cache-fit { width: min(calc(100vw - 104px), 286px) !important; max-width: min(calc(100vw - 104px), 286px) !important; }
      }
      @media (max-width: 768px) { .paper-grid { opacity: 0.35; } }
      @media (prefers-reduced-motion: reduce) { .paper-page * { animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; transition-duration: 0.001ms !important; scroll-behavior: auto !important; } }
    `}</style>
  );
}

function PaperBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden bg-[#fcfbf8]" aria-hidden="true">
      <div className="absolute inset-0 paper-grid opacity-80" />
      <div className="absolute inset-0 bg-gradient-to-b from-white/90 via-[#fcfbf8]/95 to-white" />
    </div>
  );
}

function PaperCard({ children, className = "" }) {
  return <Card className={`paper-card w-full min-w-0 max-w-full rounded-md ${className}`}>{children}</Card>;
}

function SectionHeader({ number, title, children }) {
  return (
    <div className="mb-8 max-w-4xl">
      <div className="mb-2 text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">{number}</div>
      <h2 className="paper-serif text-3xl font-semibold tracking-tight text-slate-950 md:text-4xl">{title}</h2>
      {children && <p className="paper-pretty mt-3 max-w-3xl text-base leading-7 text-slate-600">{children}</p>}
    </div>
  );
}

function PaperNav() {
  return (
    <div className="sticky top-0 z-20 border-b border-slate-200 bg-[#fcfbf8]/92 backdrop-blur">
      <nav className="mx-auto flex max-w-[1180px] items-center justify-between px-5 py-3 lg:px-8" aria-label="Primary navigation">
        <a href="#top" className={`paper-serif text-sm font-semibold text-slate-950 ${focusClass()}`}>Cache-Coherent ASMC</a>
        <div className="hidden items-center gap-1 md:flex">
          {NAV_ITEMS.map(([label, href]) => (
            <a key={href} href={`#${href}`} className={`rounded px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-100 hover:text-slate-950 ${focusClass()}`}>
              {label}
            </a>
          ))}
        </div>
      </nav>
    </div>
  );
}

function StatStrip() {
  return (
    <div className="grid gap-3 border-y border-slate-200 py-4 sm:grid-cols-2 lg:grid-cols-4">
      {STATS.map(([label, value]) => (
        <div key={label} className="px-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <div className="paper-serif text-3xl font-semibold text-slate-950">{value}</div>
            <EvidenceBadge />
          </div>
          <div className="mt-1 text-sm text-slate-500">{label}</div>
        </div>
      ))}
    </div>
  );
}

function FigureCaption({ label, children }) {
  return (
    <p className="mt-3 text-sm leading-6 text-slate-600">
      <span className="font-semibold text-slate-950">{label}.</span>{" "}
      <EvidenceBadge compact /> {children}
    </p>
  );
}

function FigureImage({ src, alt, minWidth = "min-w-[680px]" }) {
  return (
    <div className="figure-scroll overflow-x-auto rounded border border-slate-200 bg-white p-2">
      <img
        src={src}
        alt={alt}
        loading="lazy"
        className={`block h-auto w-full max-w-none select-none rounded-sm ${minWidth} sm:min-w-0 sm:max-w-full`}
      />
    </div>
  );
}

function FigurePanel({ eyebrow, title, icon, children, captionLabel, caption, href }) {
  return (
    <PaperCard className="overflow-hidden">
      <CardContent className="p-0">
        <div className="flex items-center justify-between gap-4 border-b border-slate-200 bg-slate-50/70 px-4 py-3 sm:px-5">
          <div className="min-w-0">
            <div className="paper-mono text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">{eyebrow}</div>
            <h3 className="mt-1 text-sm font-semibold leading-snug text-slate-950">{title}</h3>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {icon && <div className="hidden h-9 w-9 place-items-center rounded border border-slate-200 bg-white text-blue-700 sm:grid">{icon}</div>}
            {href && (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                aria-label={`Open ${title}`}
                className={`grid h-9 w-9 place-items-center rounded border border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50 hover:text-slate-950 ${focusClass()}`}
              >
                <ExternalLink size={16} aria-hidden="true" />
              </a>
            )}
          </div>
        </div>
        <div className="p-4 sm:p-5">
          {children}
          <FigureCaption label={captionLabel}>{caption}</FigureCaption>
        </div>
      </CardContent>
    </PaperCard>
  );
}

function EvidenceStrip({ items }) {
  return (
    <div className="mb-4 grid gap-3 md:grid-cols-3">
      {items.map((item) => (
        <div key={item.label} className="rounded border border-slate-200 bg-slate-50 px-3 py-2.5">
          <div className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{item.label}</div>
          <div className="mt-1 flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <div className="paper-serif text-2xl font-semibold tracking-tight text-slate-950">{item.value}</div>
            <EvidenceBadge compact />
          </div>
          <div className="mt-1 text-xs leading-5 text-slate-600">{item.detail}</div>
        </div>
      ))}
    </div>
  );
}

function EvidenceBadge({ kind = "paper", compact = false }) {
  const label = kind === "artifact" ? "Artifact-verified result" : kind === "reproduced" ? "Reproduced by public artifact" : "Paper-reported result";
  const tone = kind === "paper"
    ? "border-amber-200 bg-amber-50 text-amber-800"
    : "border-emerald-200 bg-emerald-50 text-emerald-800";
  return (
    <span
      className={`paper-mono inline-flex w-fit items-center rounded border px-1.5 py-0.5 text-[9px] font-semibold uppercase leading-tight tracking-[0.06em] ${tone} ${compact ? "text-[8px]" : ""}`}
    >
      {label}
    </span>
  );
}

function ArtifactStatus() {
  return (
    <PaperCard className="mb-8 border-amber-200 bg-amber-50/70">
      <CardContent className="p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.14em] text-slate-950">
            <ShieldCheck size={18} className="text-amber-700" aria-hidden="true" />
            Artifact status
          </div>
          <span className="paper-mono rounded border border-amber-200 bg-white/70 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-amber-800">
            GPU rerun pending
          </span>
        </div>
        <p className="paper-pretty mt-3 max-w-4xl text-sm leading-6 text-slate-700">
          This page reports the ICML 2026 camera-ready results. The corrected public ASMC implementation and CPU tests are available. A corrected, audited 500-problem GPU rerun is pending. See{" "}
          <a className="font-semibold text-blue-700 underline decoration-blue-300 underline-offset-2" href="https://github.com/Vicky-0256/ASMC/blob/main/docs/result_integrity.md" target="_blank" rel="noreferrer">Result Integrity</a>{" "}
          and{" "}
          <a className="font-semibold text-blue-700 underline decoration-blue-300 underline-offset-2" href="https://github.com/Vicky-0256/ASMC/blob/main/docs/reproducibility.md" target="_blank" rel="noreferrer">Reproducibility</a>{" "}
          for the exact evidence boundary.
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-2 text-xs leading-5 text-slate-600">
          <EvidenceBadge />
          <span>Current paper-facing numbers use this label until the corrected rerun is complete.</span>
        </div>
      </CardContent>
    </PaperCard>
  );
}

function MotivationSection() {
  return (
    <section id="motivation" className="py-14">
      <SectionHeader number="1. Motivation" title="Tail latency is the deployment bottleneck.">
        Chain-based MCMC can target stronger trajectory distributions, but the generate-then-verify loop is serial.
        ASMC turns temporal search into a parallel particle search.
      </SectionHeader>

      <div className="grid gap-4 lg:grid-cols-3">
        <PaperCard>
          <CardContent className="p-5">
            <div className="mb-3 grid h-10 w-10 place-items-center rounded bg-red-50 text-red-600"><Timer size={19} aria-hidden="true" /></div>
            <h3 className="font-semibold text-slate-950">Serial decode loop</h3>
            <p className="paper-pretty mt-2 text-sm leading-6 text-slate-600">MCMC proposes, scores, accepts or rejects, then repeats. Each step depends on the previous chain state.</p>
          </CardContent>
        </PaperCard>
        <PaperCard>
          <CardContent className="p-5">
            <div className="mb-3 grid h-10 w-10 place-items-center rounded bg-orange-50 text-orange-600"><Activity size={19} aria-hidden="true" /></div>
            <h3 className="font-semibold text-slate-950">GPU under-utilization</h3>
            <p className="paper-pretty mt-2 text-sm leading-6 text-slate-600">Serial control flow leaves batch parallelism on the table, especially during long autoregressive decoding.</p>
          </CardContent>
        </PaperCard>
        <PaperCard>
          <CardContent className="p-5">
            <div className="mb-3 grid h-10 w-10 place-items-center rounded bg-blue-50 text-blue-700"><Workflow size={19} aria-hidden="true" /></div>
            <h3 className="font-semibold text-slate-950">Particle inference</h3>
            <p className="paper-pretty mt-2 text-sm leading-6 text-slate-600">A population of weighted particles evolves together, enabling batched expansion and adaptive compute allocation.</p>
          </CardContent>
        </PaperCard>
      </div>

      <div className="mt-5">
        <FigurePanel
          eyebrow="Supplementary Figure"
          title="ASMC parallel particle search versus sequential MH"
          icon={<Workflow size={18} aria-hidden="true" />}
          href={figureSrc("ASMC_vs_MH.png")}
          captionLabel="Supplementary Figure"
          caption="The paper contrasts ASMC's batched particle population with MH's single-chain mutate-and-accept loop; rejection in MH wastes compute and preserves serial dependence."
        >
          <FigureImage
            src={figureSrc("ASMC_vs_MH.png")}
            alt="Algorithmic comparison of ASMC and Metropolis-Hastings sampling for reasoning."
            minWidth="min-w-[860px]"
          />
        </FigurePanel>
      </div>
    </section>
  );
}

function MethodDetailsSection() {
  return (
    <section id="demo" className="py-14">
      <SectionHeader number="2. Method details" title="How Cache-Coherent ASMC works">
        ASMC replaces a serial MCMC chain with a batched particle population.
        When resampling changes particle ancestry, the same ancestor map reorders
        particle traces, Transformer KV state, and particle-bound tensors.
      </SectionHeader>

      <div className="grid gap-5 lg:grid-cols-3">
        <PaperCard>
          <CardContent className="p-5">
            <h3 className="font-semibold text-slate-950">Parallel particles</h3>
            <p className="paper-pretty mt-2 text-sm leading-6 text-slate-600">
              N particles decode token blocks together, improving GPU utilization
              over sequential chain-based MCMC.
            </p>
          </CardContent>
        </PaperCard>

        <PaperCard>
          <CardContent className="p-5">
            <h3 className="font-semibold text-slate-950">ESS-triggered resampling</h3>
            <p className="paper-pretty mt-2 text-sm leading-6 text-slate-600">
              When ESS falls below τN, the resampling scheme selects ancestors
              and resets particle weights to 1/N.
            </p>
          </CardContent>
        </PaperCard>

        <PaperCard>
          <CardContent className="p-5">
            <h3 className="font-semibold text-slate-950">Cache-coherent reorder</h3>
            <p className="paper-pretty mt-2 text-sm leading-6 text-slate-600">
              KV caches, position IDs, attention masks, and decoding buffers are
              gathered by the same composed ancestor map.
            </p>
          </CardContent>
        </PaperCard>
      </div>

      <div className="mt-5">
        <FigurePanel
          eyebrow="Paper Figure 1"
          title="Method overview"
          icon={<Workflow size={18} aria-hidden="true" />}
          href={figureSrc("ASMC_newest.png")}
          captionLabel="Paper Figure 1"
          caption="ASMC evolves particles in parallel and applies the same ancestry map to particle state and KV cache state after resampling."
        >
          <FigureImage
            src={figureSrc("ASMC_newest.png")}
            alt="ASMC method diagram showing parallel expansion, particle weight updates, residual resampling, KV cache reorder, early stop, weighted vote, and hard pass restart."
            minWidth="min-w-[760px]"
          />
        </FigurePanel>
      </div>
    </section>
  );
}

function ResultsSection() {
  return (
    <section id="results" className="py-14">
      <ArtifactStatus />
      <SectionHeader number="3. Results" title="Accuracy improves without inheriting MCMC tail latency.">
        The deployment-facing comparison is the accuracy–p95 frontier, not just mean compute budget.
      </SectionHeader>
      <div className="mb-8"><StatStrip /></div>
      <div className="grid gap-5 lg:grid-cols-[1.35fr_0.65fr]">
        <FigurePanel
          eyebrow="Paper Figure 2"
          title="Accuracy vs. p95 latency"
          icon={<BarChart3 size={18} aria-hidden="true" />}
          href={figureSrc("acc_vs_p95.png")}
          captionLabel="Paper Figure 2"
          caption="ASMC and ASMC-adaptive occupy high-accuracy, low-tail-latency operating points, while sequential MCMC moves far to the high-p95 region."
        >
            <FigureImage
              src={figureSrc("acc_vs_p95.png")}
              alt="Accuracy versus p95 latency frontier on MATH500 comparing Naive, Best-of-N, MCMC, ASMC, and ASMC-adaptive."
              minWidth="min-w-[560px]"
            />
        </FigurePanel>
        <PaperCard>
          <CardContent className="p-5">
            <div className="mb-4 flex items-center gap-2 font-semibold text-slate-950"><Table2 size={18} aria-hidden="true" /> Selected MATH500 rows</div>
            <div className="overflow-hidden rounded border border-slate-200">
              <table className="w-full border-collapse text-sm" aria-label="Selected MATH500 latency-aware results">
                <caption className="sr-only">Selected MATH500 latency-aware results with budget, method, accuracy, p95 latency, and evidence status. The evidence label applies to every numeric value in its row.</caption>
                <thead className="bg-slate-50 text-left text-slate-600"><tr><th scope="col" className="border-b border-slate-200 px-3 py-2 font-semibold">Budget</th><th scope="col" className="border-b border-slate-200 px-3 py-2 font-semibold">Method</th><th scope="col" className="border-b border-slate-200 px-3 py-2 text-right font-semibold">Acc.</th><th scope="col" className="border-b border-slate-200 px-3 py-2 text-right font-semibold">p95</th><th scope="col" className="border-b border-slate-200 px-3 py-2 font-semibold">Evidence</th></tr></thead>
                <tbody>{RESULT_ROWS.map((row, index) => {
                  const isAdaptive = row.method.includes("ASMC-adapt");
                  const isMcmc = row.method.includes("MCMC");
                  const rowTone = isAdaptive
                    ? "bg-emerald-50/80 font-semibold text-emerald-900"
                    : isMcmc
                      ? "bg-red-50/60 text-red-900"
                      : index % 2
                        ? "bg-slate-50/70"
                        : "bg-white";

                  return (
                    <tr key={`${row.budget}-${row.method}`} className={rowTone}>
                      <td className={`px-3 py-2 paper-mono text-xs ${isAdaptive ? "border-l-2 border-emerald-500" : isMcmc ? "border-l-2 border-red-400" : "border-l-2 border-transparent"}`} aria-label={`${row.budget}, Paper-reported result`}>{row.budget}</td>
                      <td className="px-3 py-2">
                        <div className="flex min-w-0 flex-col gap-1">
                          <span>{row.method}</span>
                          {row.badge && (
                            <span className={`paper-mono w-fit rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] ${isAdaptive ? "border-emerald-200 bg-white/70 text-emerald-700" : "border-red-200 bg-white/70 text-red-700"}`}>
                              {row.badge}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-3 py-2 text-right paper-mono text-xs" aria-label={`${row.acc}% accuracy, Paper-reported result`}>{row.acc}%</td>
                      <td className="px-3 py-2 text-right paper-mono text-xs" aria-label={`${row.p95} seconds p95, Paper-reported result`}>{row.p95}s</td>
                      <td className="px-3 py-2"><EvidenceBadge compact /></td>
                    </tr>
                  );
                })}</tbody>
              </table>
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-600">Best-of-N improves quickly but saturates; sequential MCMC has the most severe tail latency.</p>
          </CardContent>
        </PaperCard>
      </div>
    </section>
  );
}

function SystemSection() {
  return (
    <section id="system" className="py-14">
      <SectionHeader number="4. System evidence" title="Cache-coherent reorder makes resampling practical.">
        The paper microbenchmarks one resampling event and compares full prefix replay with KV reorder on Qwen2.5-Math-7B.
      </SectionHeader>
      <div className="grid gap-5">
        <FigurePanel
          eyebrow="Paper Figure 3"
          title="System cost of a resampling event"
          icon={<Activity size={18} aria-hidden="true" />}
          href={figureSrc("figure2_e2.png")}
          captionLabel="Paper Figure 3"
          caption="KV reorder reduces event latency by 53–77× relative to prefix replay and remains feasible in regimes where rebuild hits timeout or OOM."
        >
            <EvidenceStrip items={SYSTEM_FACTS} />
            <FigureImage
              src={figureSrc("figure2_e2.png")}
              alt="System cost of a resampling event: event latency versus sequence length, peak allocated memory, and feasibility for repeated events."
              minWidth="min-w-[980px]"
            />
        </FigurePanel>
      </div>
    </section>
  );
}

function CollapseSection() {
  return (
    <section id="collapse" className="py-14">
      <SectionHeader number="5. Diagnostics" title="Particle collapse explains many failures.">
        Maintaining diversity matters more than fine-tuning the specific resampling scheme.
      </SectionHeader>
      <div className="grid gap-5">
        <FigurePanel
          eyebrow="Paper Figure 4"
          title="Particle collapse diagnostics"
          icon={<ShieldCheck size={18} aria-hidden="true" />}
          href={figureSrc("fig3_combined.png")}
          captionLabel="Paper Figure 4"
          caption="Frequent resampling and low ESSmin/N expose particle collapse, while residual versus multinomial resampling is secondary."
        >
            <EvidenceStrip items={COLLAPSE_FACTS} />
            <FigureImage
              src={figureSrc("fig3_combined.png")}
              alt="Particle collapse diagnostics: accuracy versus resampling frequency, ESS distribution and CDF for correct versus incorrect problems, and residual versus multinomial resampling comparison."
              minWidth="min-w-[980px]"
            />
        </FigurePanel>
        <PaperCard>
          <CardContent className="p-5">
            <div className="grid gap-4 md:grid-cols-3">{COLLAPSE_ROWS.map((row) => <div key={row.diagnostic} className="rounded border border-slate-200 bg-slate-50 p-4"><div className="mb-3 grid h-9 w-9 place-items-center rounded bg-violet-50 text-violet-700"><ShieldCheck size={18} aria-hidden="true" /></div><div className="font-semibold text-slate-950">{row.diagnostic}</div><p className="paper-pretty mt-2 text-sm leading-6 text-slate-600">{row.interpretation}</p></div>)}</div>
          </CardContent>
        </PaperCard>
      </div>
    </section>
  );
}

function TakeawaysSection() {
  return <section className="py-14"><PaperCard className="overflow-hidden"><CardContent className="grid gap-8 p-7 lg:grid-cols-[0.9fr_1.1fr] lg:items-center"><div><div className="mb-2 text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">Summary</div><h2 className="paper-serif text-3xl font-semibold tracking-tight text-slate-950 md:text-4xl">Reasoning as particle flow; resampling as cache reordering.</h2></div><div className="grid gap-3">{TAKEAWAYS.map((item) => <div key={item} className="flex items-center gap-3 rounded border border-slate-200 bg-slate-50 p-3 text-sm font-medium text-slate-700"><CheckCircle2 className="text-emerald-600" size={18} aria-hidden="true" /> {item}</div>)}</div></CardContent></PaperCard></section>;
}

export default function ASMCPaperPage() {
  return (
    <div className="paper-page min-h-screen overflow-x-hidden bg-[#fcfbf8] text-slate-950">
      <DesignSystemStyles />
      <PaperBackground />
      <PaperNav />
      <main className="relative mx-auto min-w-0 max-w-[1180px] px-5 pb-16 lg:px-8">
        <HeroWithModelAnimation />
        <section id="abstract" className="paper-rule py-10"><div className="grid gap-6 lg:grid-cols-[180px_1fr]"><div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.18em] text-slate-500"><BookOpen size={16} aria-hidden="true" /> Abstract</div><p className="paper-pretty text-lg leading-8 text-slate-700">Test-time scaling can improve LLM reasoning without retraining, but sequential trajectory samplers suffer from poor GPU utilization and severe tail latency. ASMC uses a parallel particle population to approximate power-shaped trajectory distributions, while cache-coherent resampling keeps Transformer state consistent by reordering KV caches and particle-bound tensors under the ancestor map.</p></div></section>
        <MotivationSection />
        <MethodDetailsSection />
        <ResultsSection />
        <SystemSection />
        <CollapseSection />
        <TakeawaysSection />
      </main>
    </div>
  );
}

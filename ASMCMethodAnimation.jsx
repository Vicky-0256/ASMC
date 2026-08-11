import React, { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import {
  Play,
  Pause,
  RotateCcw,
  Layers,
  CheckCircle2,
  AlertTriangle,
  SkipForward,
  Repeat2,
  GitBranch,
  Github,
  BookOpen,
  Copy,
  ChevronRight,
  ArrowRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const DELTA = 0.65;
const DEFAULT_N_HARD = 16;
const DEFAULT_THRESHOLD = 0.5;
const DEFAULT_SPEED = 1300;
const DEFAULT_DEMO_MODE = "resample";
const BUDGET_CAP = 1.0;

const STEPS = [
  { key: "prompt", label: "Prompt", short: "Prompt", note: "Create a parallel particle population from the prompt." },
  { key: "expand", label: "Expand", short: "Expand", note: "All particles decode the next token block together on GPU." },
  { key: "reweight", label: "Reweight + ESS", short: "Weight", note: "Weights target the fixed α⋆ trajectory distribution while the proposal anneals with βt; ESS checks collapse." },
  { key: "resample", label: "Resample / Skip", short: "Sample", note: "Particles are resampled by normalized weights only when ESS drops below the threshold." },
  { key: "reorder", label: "KV Reorder", short: "Cache", note: "When ancestry changes, gather KV state slices and particle-bound tensors instead of replaying prefixes." },
  { key: "resolve", label: "Continue / Vote", short: "Resolve", note: "After resampling, continue to the next block; guarded top-answer mass decides output or a fresh hard pass." },
];

const LANES = [
  { id: "P1", weight: 0.36, color: "#1f77b4", tokens: ["Let", "x", "=", "..."] },
  { id: "P2", weight: 0.08, color: "#7f7f7f", tokens: ["Try", "case", "1", "..."] },
  { id: "P3", weight: 0.42, color: "#6a3d9a", tokens: ["Since", "sum", "is", "..."] },
  { id: "P4", weight: 0.14, color: "#2ca02c", tokens: ["First", "solve", "for", "..."] },
];

const ANCESTOR_MAP = [3, 1, 3, 4];
const CACHE_TOKENS = [0, 1, 2, 3, 4];
const CACHE_COLORS = {
  1: "#1f77b4",
  2: "#7f7f7f",
  3: "#6a3d9a",
  4: "#2ca02c",
};

const DEMO_MODES = [
  { value: "resample", label: "Low-ESS / resampling" },
  { value: "easy", label: "Easy / early vote" },
  { value: "hard", label: "Hard restart" },
  { value: "budget", label: "Budget cap" },
];


function getDemoSignals(step, mode) {
  const passBoundary = step >= 5;

  // These deterministic signals are only for the visual storyboard; they let one UI demonstrate early exit, resampling, hard restart, and budget cap without running a model.
  if (mode === "easy") {
    return {
      ess: 0.88,
      topMassDuringBlock: step >= 2 ? 0.72 : 0.26,
      topMassAtBoundary: 0.72,
      passBoundary,
      budgetUsed: Math.min(0.76, 0.12 + step * 0.12),
    };
  }

  if (mode === "hard") {
    return {
      ess: 0.72,
      topMassDuringBlock: step >= 2 ? 0.42 : 0.26,
      topMassAtBoundary: 0.42,
      passBoundary,
      budgetUsed: Math.min(0.86, 0.16 + step * 0.13),
    };
  }

  if (mode === "budget") {
    return {
      ess: 0.74,
      topMassDuringBlock: step >= 2 ? 0.44 : 0.26,
      topMassAtBoundary: 0.44,
      passBoundary,
      budgetUsed: Math.min(BUDGET_CAP, 0.3 + step * 0.18),
    };
  }

  return {
    ess: step < 2 ? 0.88 : 0.42,
    topMassDuringBlock: step >= 2 ? 0.42 : 0.26,
    topMassAtBoundary: 0.42,
    passBoundary,
    budgetUsed: Math.min(0.9, 0.14 + step * 0.12),
  };
}

function getRuntimeState(step, threshold, nHard, mode) {
  const nFast = Math.max(4, Math.floor(nHard / 2));
  const signals = getDemoSignals(step, mode);
  const needResample = signals.ess < threshold;
  const weightsReset = needResample && step >= 4;
  const continueNextBlock = step >= 5 && weightsReset;
  const passBoundary = signals.passBoundary && !weightsReset;
  const budgetReached = signals.budgetUsed >= BUDGET_CAP;

  // The demo visualizes the fast pass. In hard mode, the decision starts a fresh hard pass from the prompt with Nhard, rather than mutating the current fast-pass particles.
  const currentPass = "fast";
  const currentN = nFast;
  const topMass = weightsReset ? 1 / currentN : passBoundary ? signals.topMassAtBoundary : signals.topMassDuringBlock;
  const confident = topMass >= DELTA;
  const earlyExit = mode === "easy" && step >= 2 && confident && !budgetReached;
  const shouldTerminate = budgetReached;
  const shouldOutput = !shouldTerminate && (earlyExit || (passBoundary && confident));
  const shouldRestart = passBoundary && !confident && !shouldTerminate;
  const currentEffectiveN = currentN;
  const nextN = shouldRestart ? nHard : currentN;

  let status = "Initializing particles";
  switch (step) {
    case 1:
      status = "Batched decoding";
      break;
    case 2:
      status = shouldTerminate ? "Budget cap reached" : earlyExit ? "Early exit possible" : "Computing weights + ESS";
      break;
    case 3:
      status = shouldTerminate ? "Budget cap reached" : needResample ? "Resampling triggered" : "Resampling skipped";
      break;
    case 4:
      status = shouldTerminate ? "Budget cap reached" : needResample ? "KV state gathered" : "KV state carried forward";
      break;
    case 5:
      status = shouldTerminate ? "Budget cap reached" : continueNextBlock ? "Continue next block" : shouldOutput ? "Weighted vote ready" : "Fast-pass restart decision";
      break;
    default:
      status = shouldTerminate ? "Budget cap reached" : "Initializing particles";
  }

  return {
    step,
    nFast,
    nHard,
    currentPass,
    currentN,
    currentEffectiveN,
    nextN,
    ess: signals.ess,
    needResample,
    weightsReset,
    continueNextBlock,
    passBoundary,
    topMass,
    confident,
    earlyExit,
    shouldOutput,
    shouldRestart,
    shouldTerminate,
    budgetUsed: signals.budgetUsed,
    budgetReached,
    status,
    mode,
  };
}

function getStepLabel(stepItem, runtimeState, isCurrent = false) {
  if (runtimeState.shouldTerminate && isCurrent) {
    return { ...stepItem, label: "Terminate", short: "Cap", note: "The per-instance compute cap is reached, so inference terminates immediately." };
  }
  if (runtimeState.earlyExit && isCurrent) {
    return { ...stepItem, label: "Early vote", short: "Early", note: "Top-mass confidence exceeds δ during the Fast pass, so ASMC can return early." };
  }
  if (runtimeState.step >= 2 && stepItem.key === "resample" && !runtimeState.needResample) {
    return { ...stepItem, label: "Skip", short: "Skip", note: "ESS is healthy enough, so no particles are copied and no ancestry changes." };
  }
  if (runtimeState.step >= 2 && stepItem.key === "reorder" && !runtimeState.needResample) {
    return { ...stepItem, label: "Carry", short: "Carry", note: "No ancestry change means KV state can be carried forward as-is." };
  }
  if (stepItem.key === "resolve" && runtimeState.continueNextBlock) {
    return { ...stepItem, label: "Continue", short: "Continue", note: "After resampling, weights reset to 1/N and ASMC continues to the next token block." };
  }
  if (stepItem.key === "resolve" && runtimeState.shouldRestart) {
    return { ...stepItem, label: "Restart", short: "Restart", note: "Fast pass ended with low confidence: set N ← Nhard and restart the run/block." };
  }
  if (stepItem.key === "resolve" && runtimeState.shouldOutput) {
    return {
      ...stepItem,
      label: runtimeState.earlyExit ? "Early vote" : "Vote",
      short: runtimeState.earlyExit ? "Early" : "Vote",
      note: runtimeState.earlyExit ? "Top-mass confidence exceeds δ during the Fast pass, so ASMC can return early." : "Confidence is high enough to output the selected answer.",
    };
  }
  return stepItem;
}

function getMainStatusInfo(runtimeState, threshold) {
  if (runtimeState.shouldTerminate) {
    return {
      tone: "border-red-200 bg-red-50 text-red-700",
      title: "Budget cap reached",
      detail: `Demo Cint = ${runtimeState.budgetUsed.toFixed(2)} ≥ C* = ${BUDGET_CAP.toFixed(2)}; inference terminates immediately.`,
    };
  }

  if (runtimeState.earlyExit || runtimeState.shouldOutput) {
    return {
      tone: "border-emerald-200 bg-emerald-50 text-emerald-700",
      title: runtimeState.earlyExit ? "Early exit possible" : "Weighted vote ready",
      detail: `Top mass = ${runtimeState.topMass.toFixed(2)} ≥ δ = ${DELTA.toFixed(2)}; later steps are bypassed.`,
    };
  }

  if (runtimeState.shouldRestart) {
    return {
      tone: "border-orange-200 bg-orange-50 text-orange-700",
      title: "Fast-pass decision: restart",
      detail: `Fast pass ended below confidence; next run uses Nhard = ${runtimeState.nHard}.`,
    };
  }

  if (runtimeState.continueNextBlock) {
    return {
      tone: "border-blue-200 bg-blue-50 text-blue-700",
      title: "Continue next block",
      detail: "Weights are reset to 1/N and both particle traces and KV state now follow the same ancestor map.",
    };
  }

  if (runtimeState.weightsReset) {
    return {
      tone: "border-blue-200 bg-blue-50 text-blue-700",
      title: "Cache-coherent state inherited",
      detail: "Resampled particles have reset weights; trajectory state and KV state are already aligned by ancestry.",
    };
  }

  if (runtimeState.needResample) {
    return {
      tone: "border-orange-200 bg-orange-50 text-orange-700",
      title: "Need resampling",
      detail: `ESS / N = ${runtimeState.ess.toFixed(2)} < τ = ${threshold.toFixed(2)}; low-ESS particles will be resampled.`,
    };
  }

  return {
    tone: "border-emerald-200 bg-emerald-50 text-emerald-700",
    title: "Skip resampling",
    detail: `ESS / N = ${runtimeState.ess.toFixed(2)} ≥ τ = ${threshold.toFixed(2)}; no ancestry change, so KV state is carried forward.`,
  };
}

function getPathInfo(runtimeState) {
  if (runtimeState.step < 2) return "Prompt / expand → awaiting ESS check";
  if (runtimeState.shouldTerminate) return "Budget cap → terminate";
  if (runtimeState.earlyExit) return "Easy path → early vote";
  if (runtimeState.shouldRestart) return "Fast pass end → fresh hard pass from prompt";
  if (runtimeState.continueNextBlock) return "Low-ESS → resample → compose ancestry → KV gather → continue";
  if (runtimeState.weightsReset) return "Low-ESS → compose ancestry → KV gather";
  if (runtimeState.needResample) return "Low-ESS → resample required";
  return "Healthy ESS → skip resampling → carry state";
}


function focusClass() {
  return "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 focus-visible:ring-offset-2 focus-visible:ring-offset-[#fcfbf8]";
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


function FigureCaption({ label, children }) {
  return (
    <p className="mt-3 text-sm leading-6 text-slate-600">
      <span className="font-semibold text-slate-950">{label}.</span> {children}
    </p>
  );
}


function ProcessStepper({ step, setStep, runtimeState, onManualStep }) {
  const terminalLocked = runtimeState.shouldTerminate || runtimeState.earlyExit || runtimeState.shouldOutput || runtimeState.shouldRestart;

  return (
    <PaperCard>
      <CardContent className="p-3">
        <div className="flex items-center gap-2 overflow-x-auto pb-1" role="tablist" aria-label="ASMC animation steps">
          {STEPS.map((rawItem, index) => {
            const active = index === step;
            const done = index < step;
            const futureBypassed = terminalLocked && index > step;
            const item = futureBypassed
              ? { ...rawItem, label: "Bypassed", short: "Bypass", note: "Later steps are bypassed after a terminal decision." }
              : getStepLabel(rawItem, runtimeState, active);
            const bypassed = !runtimeState.needResample && (rawItem.key === "resample" || rawItem.key === "reorder");
            const restart = rawItem.key === "resolve" && runtimeState.shouldRestart;
            const continueBlock = rawItem.key === "resolve" && runtimeState.continueNextBlock;
            const terminate = runtimeState.shouldTerminate && active;

            return (
              <div key={rawItem.key} className="flex shrink-0 items-center gap-2">
                <button
                  type="button"
                  role="tab"
                  aria-selected={active}
                  aria-disabled={futureBypassed}
                  disabled={futureBypassed}
                  onClick={() => {
                    if (futureBypassed) return;
                    onManualStep();
                    setStep(index);
                  }}
                  className={`flex items-center gap-2 rounded border px-3 py-2 text-sm transition ${focusClass()} ${
                    futureBypassed
                      ? "cursor-not-allowed border-slate-200 bg-slate-50 text-slate-400"
                      : active
                        ? terminate || restart
                          ? "border-orange-300 bg-orange-50 text-orange-800"
                          : bypassed || continueBlock
                            ? "border-emerald-300 bg-emerald-50 text-emerald-800"
                            : "border-slate-900 bg-slate-900 text-white"
                        : done
                          ? "border-emerald-200 bg-emerald-50 text-slate-700 hover:bg-emerald-100"
                          : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  <span className="paper-mono text-xs">{index + 1}</span>
                  <span className="whitespace-nowrap font-medium">{item.short}</span>
                  {restart && <Repeat2 size={13} aria-hidden="true" />}
                  {(bypassed || futureBypassed) && <SkipForward size={13} aria-hidden="true" />}
                </button>
                {index < STEPS.length - 1 && <ChevronRight size={15} className="text-slate-300" aria-hidden="true" />}
              </div>
            );
          })}
        </div>
      </CardContent>
    </PaperCard>
  );
}

function RuntimeStateBar({ step, threshold, runtimeState }) {
  const currentStep = getStepLabel(STEPS[step], runtimeState, true);

  return (
    <PaperCard>
      <CardContent className="p-4">
        <div className="grid gap-4 xl:grid-cols-[1.05fr_0.7fr_0.75fr_0.85fr_0.85fr_0.85fr_0.9fr] xl:items-center">
          <div>
            <div className="mb-1 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Runtime State</div>
            <div className="text-lg font-semibold text-slate-950">{currentStep.label}</div>
            <div className="paper-pretty mt-1 text-xs leading-relaxed text-slate-500">{currentStep.note}</div>
          </div>
          <div className="rounded border border-slate-200 bg-slate-50 px-4 py-3">
            <div className="text-xs text-slate-500">Pass</div>
            <div className="mt-1 text-sm font-semibold capitalize text-slate-900">{runtimeState.currentPass}</div>
          </div>
          <div className="rounded border border-emerald-200 bg-emerald-50 px-4 py-3">
            <div className="text-xs text-emerald-700/70">Status</div>
            <div className="mt-1 text-sm font-semibold text-emerald-800">{runtimeState.status}</div>
          </div>
          <div>
            <div className="mb-2 flex justify-between text-xs text-slate-500"><span>ESS / N</span><span>{runtimeState.ess.toFixed(2)} / τ {threshold.toFixed(2)}</span></div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-200"><motion.div className={`h-full rounded-full ${runtimeState.needResample ? "bg-orange-500" : "bg-blue-600"}`} animate={{ width: `${runtimeState.ess * 100}%` }} transition={{ duration: 0.6 }} /></div>
          </div>
          <div>
            <div className="mb-2 flex justify-between text-xs text-slate-500"><span>Top-answer mass</span><span>{runtimeState.topMass.toFixed(2)} / δ {DELTA.toFixed(2)}</span></div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-200"><motion.div className={`h-full rounded-full ${runtimeState.confident ? "bg-emerald-600" : "bg-violet-700"}`} animate={{ width: `${Math.min(1, runtimeState.topMass) * 100}%` }} transition={{ duration: 0.6 }} /></div>
          </div>
          <div>
            <div className="mb-2 flex justify-between text-xs text-slate-500"><span>Demo Cint / C*</span><span>{runtimeState.budgetUsed.toFixed(2)} / {BUDGET_CAP.toFixed(2)}</span></div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-200"><motion.div className={`h-full rounded-full ${runtimeState.budgetReached ? "bg-red-600" : "bg-slate-700"}`} animate={{ width: `${Math.min(1, runtimeState.budgetUsed / BUDGET_CAP) * 100}%` }} transition={{ duration: 0.6 }} /></div>
            <div className="mt-1 text-[10px] text-slate-500">visual proxy</div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded border border-slate-200 bg-slate-50 p-3"><div className="text-xs text-slate-500">Current N</div><div className="mt-1 paper-mono text-lg font-semibold text-slate-950">{runtimeState.currentEffectiveN}</div></div>
            <div className="rounded border border-slate-200 bg-slate-50 p-3"><div className="text-xs text-slate-500">Next N</div><div className="mt-1 paper-mono text-lg font-semibold text-slate-950">{runtimeState.nextN}</div></div>
          </div>
        </div>
      </CardContent>
    </PaperCard>
  );
}

function RuntimeChip({ label, value, tone = "slate" }) {
  const tones = {
    slate: "border-slate-200 bg-slate-50 text-slate-700",
    blue: "border-blue-200 bg-blue-50 text-blue-700",
    emerald: "border-emerald-200 bg-emerald-50 text-emerald-700",
    orange: "border-orange-200 bg-orange-50 text-orange-700",
    red: "border-red-200 bg-red-50 text-red-700",
  };

  return (
    <span className={`inline-flex max-w-full items-center gap-1.5 rounded border px-2.5 py-1 text-[11px] ${tones[tone]}`}>
      <span className="text-slate-500">{label}</span>
      <span className="paper-mono min-w-0 font-semibold">{value}</span>
    </span>
  );
}

function HeroControlStrip({
  isPlaying,
  setIsPlaying,
  resetAllControls,
  stepForward,
  nHard,
  setNHard,
  threshold,
  setThreshold,
  speed,
  setSpeed,
  mode,
  setMode,
  step,
  setStep,
  runtimeState,
  onManualStep,
  terminalLocked,
  variant = "wide",
}) {
  const currentStep = getStepLabel(STEPS[step], runtimeState, true);
  const hardTerminal = runtimeState.shouldTerminate || runtimeState.earlyExit || runtimeState.shouldOutput || runtimeState.shouldRestart;
  const statusTone = runtimeState.shouldTerminate
    ? "red"
    : runtimeState.shouldRestart || runtimeState.needResample
      ? "orange"
      : runtimeState.shouldOutput || runtimeState.earlyExit || runtimeState.continueNextBlock
        ? "emerald"
        : "blue";

  return (
    <PaperCard className="min-w-0 overflow-hidden">
      <CardContent className="p-3 md:p-4">
        <div className={variant === "side" ? "grid gap-3" : "grid gap-3 xl:grid-cols-[auto_1fr_auto] xl:items-center"}>
          <div className={variant === "side" ? "grid grid-cols-3 gap-2" : "flex flex-wrap items-center gap-2"}>
            <Button
              className={`rounded bg-slate-900 text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 ${focusClass()}`}
              disabled={terminalLocked}
              onClick={() => setIsPlaying((value) => !value)}
            >
              {isPlaying ? <Pause className="mr-2 h-4 w-4" aria-hidden="true" /> : <Play className="mr-2 h-4 w-4" aria-hidden="true" />}
              {isPlaying ? "Pause" : "Play"}
            </Button>
            <Button
              variant="outline"
              className={`rounded border-slate-300 bg-white text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 ${focusClass()}`}
              disabled={terminalLocked}
              onClick={stepForward}
            >
              <SkipForward className="mr-2 h-4 w-4" aria-hidden="true" />
              Step
            </Button>
            <Button
              variant="outline"
              className={`rounded border-slate-300 bg-white text-slate-700 hover:bg-slate-50 ${focusClass()}`}
              onClick={resetAllControls}
            >
              <RotateCcw className="mr-2 h-4 w-4" aria-hidden="true" />
              Reset
            </Button>
          </div>

          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="paper-mono rounded bg-slate-900 px-2 py-1 text-xs font-semibold text-white">{step + 1}/{STEPS.length}</span>
              <span className="font-semibold text-slate-950">{currentStep.label}</span>
              <span className="min-w-0 basis-full text-xs leading-5 text-slate-500 xl:basis-auto">{currentStep.note}</span>
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              <RuntimeChip label="Status" value={runtimeState.status} tone={statusTone} />
              <RuntimeChip label="ESS/N" value={`${runtimeState.ess.toFixed(2)} / tau ${threshold.toFixed(2)}`} tone={runtimeState.needResample ? "orange" : "blue"} />
              <RuntimeChip label="Top mass" value={`${runtimeState.topMass.toFixed(2)} / delta ${DELTA.toFixed(2)}`} tone={runtimeState.confident ? "emerald" : "slate"} />
              <RuntimeChip label="N" value={`${runtimeState.currentEffectiveN} -> ${runtimeState.nextN}`} />
            </div>
          </div>

          <details className={`rounded border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600 ${variant === "side" ? "w-full" : "xl:w-[250px]"}`}>
            <summary className={`cursor-pointer select-none font-semibold text-slate-700 ${focusClass()}`}>Parameters</summary>
            <div className="mt-3 grid gap-3">
              <label className="grid gap-1.5">
                <span className="text-xs text-slate-500">Problem mode</span>
                <select
                  aria-label="Problem mode"
                  className={`w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-sm ${focusClass()}`}
                  value={mode}
                  onChange={(event) => setMode(event.target.value)}
                >
                  {DEMO_MODES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
                </select>
              </label>
              <label className="grid gap-1.5">
                <span className="flex justify-between text-xs text-slate-500">
                  <span>Nhard</span>
                  <span className="paper-mono">{nHard}</span>
                </span>
                <input aria-label="Particles Nhard" className={`w-full accent-slate-900 ${focusClass()}`} type="range" min="4" max="128" step="4" value={nHard} onChange={(event) => setNHard(Number(event.target.value))} />
              </label>
              <label className="grid gap-1.5">
                <span className="flex justify-between text-xs text-slate-500">
                  <span>ESS tau</span>
                  <span className="paper-mono">{threshold.toFixed(2)}</span>
                </span>
                <input aria-label="ESS threshold tau" className={`w-full accent-slate-900 ${focusClass()}`} type="range" min="0.3" max="0.8" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
              </label>
              <label className="grid gap-1.5">
                <span className="flex justify-between text-xs text-slate-500">
                  <span>Speed</span>
                  <span className="paper-mono">{speed}ms</span>
                </span>
                <input aria-label="Animation speed" className={`w-full accent-slate-900 ${focusClass()}`} type="range" min="800" max="2200" step="100" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} />
              </label>
            </div>
          </details>
        </div>

        <div className={variant === "side" ? "mt-3 grid grid-cols-3 gap-2 border-t border-slate-200 pt-3" : "mt-3 flex items-center gap-2 overflow-x-auto border-t border-slate-200 pt-3"} role="tablist" aria-label="ASMC animation steps">
          {STEPS.map((rawItem, index) => {
            const active = index === step;
            const done = index < step;
            const futureBypassed = hardTerminal && index > step;
            const item = futureBypassed
              ? { ...rawItem, label: "Bypassed", short: "Bypass", note: "Later steps are bypassed after a terminal decision." }
              : getStepLabel(rawItem, runtimeState, active);
            const bypassed = !runtimeState.needResample && (rawItem.key === "resample" || rawItem.key === "reorder");
            const restart = rawItem.key === "resolve" && runtimeState.shouldRestart;
            const continueBlock = rawItem.key === "resolve" && runtimeState.continueNextBlock;
            const terminate = runtimeState.shouldTerminate && active;

            return (
              <button
                key={rawItem.key}
                type="button"
                role="tab"
                aria-selected={active}
                aria-disabled={futureBypassed}
                disabled={futureBypassed}
                onClick={() => {
                  if (futureBypassed) return;
                  onManualStep();
                  setStep(index);
                }}
                className={`flex shrink-0 items-center justify-center gap-2 rounded border px-2.5 py-1.5 text-xs transition ${focusClass()} ${
                  futureBypassed
                    ? "cursor-not-allowed border-slate-200 bg-slate-50 text-slate-400"
                    : active
                      ? terminate || restart
                        ? "border-orange-300 bg-orange-50 text-orange-800"
                        : bypassed || continueBlock
                          ? "border-emerald-300 bg-emerald-50 text-emerald-800"
                          : "border-slate-900 bg-slate-900 text-white"
                      : done
                        ? "border-emerald-200 bg-emerald-50 text-slate-700 hover:bg-emerald-100"
                        : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                }`}
              >
                <span className="paper-mono">{index + 1}</span>
                <span className="whitespace-nowrap font-medium">{item.short}</span>
                {restart && <Repeat2 size={12} aria-hidden="true" />}
                {(bypassed || futureBypassed) && <SkipForward size={12} aria-hidden="true" />}
              </button>
            );
          })}
        </div>
      </CardContent>
    </PaperCard>
  );
}

function ControlPanel({ isPlaying, setIsPlaying, resetAllControls, stepForward, nHard, setNHard, threshold, setThreshold, speed, setSpeed, mode, setMode, terminalLocked }) {
  return (
    <PaperCard>
      <CardContent className="p-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex flex-wrap items-center gap-2">
            <Button className={`rounded bg-slate-900 text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 ${focusClass()}`} disabled={terminalLocked} onClick={() => setIsPlaying((value) => !value)}>{isPlaying ? <Pause className="mr-2 h-4 w-4" aria-hidden="true" /> : <Play className="mr-2 h-4 w-4" aria-hidden="true" />}{isPlaying ? "Pause" : "Play"}</Button>
            <Button variant="outline" className={`rounded border-slate-300 bg-white text-slate-700 hover:bg-slate-50 ${focusClass()}`} onClick={resetAllControls}><RotateCcw className="mr-2 h-4 w-4" aria-hidden="true" /> Reset</Button>
            <Button variant="outline" className={`rounded border-slate-300 bg-white text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 ${focusClass()}`} disabled={terminalLocked} onClick={stepForward}>Step Forward</Button>
          </div>
          <div className="grid gap-3 text-sm text-slate-600 sm:grid-cols-2 lg:w-[760px] xl:grid-cols-4">
            <label className="rounded border border-slate-200 bg-slate-50 p-3"><div className="mb-2 flex justify-between text-xs text-slate-500"><span>Problem mode</span></div><select aria-label="Problem mode" className={`w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-sm ${focusClass()}`} value={mode} onChange={(event) => setMode(event.target.value)}>{DEMO_MODES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
            <label className="rounded border border-slate-200 bg-slate-50 p-3"><div className="mb-2 flex justify-between text-xs text-slate-500"><span>Particles Nhard</span><span className="paper-mono">{nHard}</span></div><input aria-label="Particles Nhard" className={`w-full accent-slate-900 ${focusClass()}`} type="range" min="4" max="128" step="4" value={nHard} onChange={(event) => setNHard(Number(event.target.value))} /></label>
            <label className="rounded border border-slate-200 bg-slate-50 p-3"><div className="mb-2 flex justify-between text-xs text-slate-500"><span>ESS τ</span><span className="paper-mono">{threshold.toFixed(2)}</span></div><input aria-label="ESS threshold tau" className={`w-full accent-slate-900 ${focusClass()}`} type="range" min="0.3" max="0.8" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label>
            <label className="rounded border border-slate-200 bg-slate-50 p-3"><div className="mb-2 flex justify-between text-xs text-slate-500"><span>Speed</span><span className="paper-mono">{speed}ms</span></div><input aria-label="Animation speed" className={`w-full accent-slate-900 ${focusClass()}`} type="range" min="800" max="2200" step="100" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} /></label>
          </div>
        </div>
      </CardContent>
    </PaperCard>
  );
}

function PromptCell({ active, reducedMotion }) {
  return (
    <motion.div className={`asmc-mobile-fit flex min-w-0 items-center gap-3 rounded border px-3 py-2 ${active ? "border-blue-300 bg-blue-50" : "border-slate-200 bg-white"}`} style={{ width: "100%", maxWidth: "calc(100vw - 72px)" }} animate={active && !reducedMotion ? { scale: [1, 1.01, 1] } : { scale: 1 }} transition={{ duration: 0.6 }}>
      <div className="grid h-9 w-9 shrink-0 place-items-center rounded bg-blue-100 text-blue-700"><GitBranch size={18} aria-hidden="true" /></div>
      <div><div className="text-sm font-semibold text-slate-950">Prompt</div><div className="text-xs text-slate-600">Fresh fast or hard pass starts here</div></div>
    </motion.div>
  );
}

function FormulaCard({ show }) {
  if (!show) return null;
  return (
    <motion.div className="rounded border border-slate-200 bg-slate-50 p-3" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
      <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Target / proposal / weight update</div>
      <div className="grid gap-2 text-[11px] leading-relaxed text-slate-700 lg:grid-cols-3">
        <div className="rounded border border-slate-200 bg-white p-2 paper-mono">π<sup>⋆</sup><sub>t</sub>(x<sub>1:t</sub>|c) ∝ p<sub>θ</sub>(x<sub>1:t</sub>|c)<sup>α⋆</sup></div>
        <div className="rounded border border-slate-200 bg-white p-2 paper-mono">q<sub>t</sub> ∝ p<sub>θ</sub>(x<sub>t</sub>|prefix)<sup>βt</sup></div>
        <div className="rounded border border-slate-200 bg-white p-2 paper-mono">w ← w · π<sup>⋆</sup><sub>t</sub>(x<sub>1:t</sub>|c) / q<sub>t</sub>(x<sub>t</sub>|prefix)</div>
      </div>
    </motion.div>
  );
}

function TokenStrip({ tokens, removed, delay = 0, reducedMotion }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      <AnimatePresence>
        {tokens.map((token, tokenIndex) => (
          <motion.div key={`${token}-${tokenIndex}`} className="paper-mono rounded border border-slate-200 bg-slate-50 px-2 py-1 text-[11px] text-slate-600" initial={{ opacity: 0, x: -8, scale: 0.96 }} animate={{ opacity: removed ? 0.25 : 1, x: 0, scale: 1 }} exit={{ opacity: 0 }} transition={{ delay: reducedMotion ? 0 : delay + tokenIndex * 0.05, duration: 0.2 }}>{token}</motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

function CacheBlocks({ source }) {
  const color = CACHE_COLORS[source] ?? "#7f7f7f";
  return <div className="asmc-cache-fit grid w-full min-w-0 max-w-full grid-cols-5 gap-1">{CACHE_TOKENS.map((token) => <div key={token} className="h-7 min-w-0 rounded-sm border border-slate-200" style={{ background: `linear-gradient(135deg, ${color}dd, ${color}30)` }} />)}</div>;
}

function LanePairRow({ lane, index, step, runtimeState, reducedMotion, committedAncestry }) {
  const showTokens = step >= 1;
  const showWeights = step >= 2;
  const previewResampling = step === 3 && runtimeState.needResample;
  const showNewAncestry = step >= 4 && runtimeState.needResample;
  const activeAncestry = showNewAncestry ? ANCESTOR_MAP : committedAncestry;
  const hasInheritedState = Boolean(activeAncestry);
  const sourceIndex = hasInheritedState ? activeAncestry[index] : index + 1;
  const source = LANES[sourceIndex - 1] ?? lane;
  const removed = previewResampling && lane.id === "P2";
  const duplicated = previewResampling && lane.id === "P3";
  const useEqualWeight = runtimeState.weightsReset || (hasInheritedState && step < 2);
  const visualWeight = useEqualWeight ? 1 / runtimeState.currentN : source.weight;
  const weightLabel = useEqualWeight ? "1/N" : source.weight.toFixed(2);
  const mappingLabel = showNewAncestry
    ? `New P${index + 1} ← Old ${source.id}`
    : hasInheritedState
      ? `P${index + 1} carries Old ${source.id}`
      : `P${index + 1} ↔ KV P${index + 1}`;
  const mappingSubLabel = showNewAncestry
    ? "composed ancestry for trace + state"
    : hasInheritedState
      ? "committed ancestry from previous block"
      : "same visual particle lane";

  // The left trace and right KV visualization intentionally use the same composed ancestry after resampling to show cache-coherent state inheritance.
  return (
    <motion.div layout className="asmc-mobile-fit grid w-full min-w-0 items-center gap-3 overflow-hidden rounded border border-slate-200 bg-white p-3 md:grid-cols-[1.15fr_120px_0.95fr]" style={{ width: "100%", maxWidth: "calc(100vw - 72px)" }} initial={{ opacity: 0, y: 8 }} animate={{ opacity: removed ? 0.45 : 1, y: 0, scale: duplicated && !reducedMotion ? [1, 1.006, 1] : 1 }} transition={{ delay: reducedMotion ? 0 : index * 0.04, duration: 0.28 }}>
      <div className="min-w-0">
        <div className="mb-2 flex items-center gap-2">
          <motion.div className="grid shrink-0 place-items-center rounded-full font-bold text-white" style={{ backgroundColor: source.color }} animate={{ width: showWeights ? 28 + visualWeight * 32 : 30, height: showWeights ? 28 + visualWeight * 32 : 30 }} transition={{ duration: 0.45 }}><span className="paper-mono text-[10px]">P{index + 1}</span></motion.div>
          <div className="paper-mono text-xs font-medium text-slate-700">{hasInheritedState ? `trajectory ← ${source.id}` : lane.id}</div>
          {duplicated && <span className="rounded-full bg-slate-900 px-1.5 py-0.5 text-[10px] text-white">×2</span>}
          <div className="ml-auto hidden w-20 overflow-hidden rounded-full bg-slate-200 sm:block"><motion.div className="h-2 rounded-full" style={{ backgroundColor: source.color }} animate={{ width: showWeights ? `${Math.round(visualWeight * 100)}%` : "18%" }} transition={{ duration: 0.5 }} /></div>
          <div className="paper-mono w-9 text-right text-xs text-slate-600">{showWeights ? weightLabel : ""}</div>
        </div>
        {showTokens ? <TokenStrip tokens={source.tokens} removed={removed} reducedMotion={reducedMotion} /> : <div className="text-xs text-slate-500">waiting for batched decode</div>}
      </div>
      <div className={`min-w-0 rounded border px-2 py-2 text-center text-[11px] font-semibold ${showNewAncestry ? "border-blue-200 bg-blue-50 text-blue-700" : "border-slate-200 bg-slate-50 text-slate-600"}`}><div className="paper-mono break-words">{mappingLabel}</div><div className="mt-1 text-[10px] font-normal text-slate-500">{mappingSubLabel}</div></div>
      <div className="min-w-0"><div className="mb-2 flex min-w-0 items-center justify-between gap-2"><div className="paper-mono min-w-0 truncate text-xs font-medium text-slate-700">{hasInheritedState ? `KV state slice ← Old ${source.id}` : `KV state slice ${lane.id}`}</div><div className="shrink-0 text-[10px] text-slate-500">KV + tensors</div></div><CacheBlocks source={sourceIndex} /></div>
    </motion.div>
  );
}

function IntegratedMethodFigure({ step, runtimeState, threshold, reducedMotion, committedAncestry }) {
  const correspondenceActive = step >= 4 && runtimeState.needResample;
  const statusInfo = getMainStatusInfo(runtimeState, threshold);
  const pathInfo = getPathInfo(runtimeState);

  return (
    <PaperCard className="min-w-0 overflow-hidden">
      <CardContent className="min-w-0 overflow-hidden p-4 sm:p-5">
        <div className="mb-4 flex min-w-0 flex-wrap items-start justify-between gap-3"><div className="min-w-0"><div className="text-sm font-semibold text-slate-950">ASMC main loop + KV state</div><div className="text-xs text-slate-500"><span className="sm:hidden">Particle lanes and KV slices stay aligned after resampling.</span><span className="hidden sm:inline">Each row shows the same particle lane on the left and its bound KV state slice plus particle-bound tensors on the right.</span></div><div className="mt-2 flex min-w-0 flex-wrap gap-2 text-[11px]"><span className="max-w-full rounded border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600">Current path: {pathInfo}</span><span className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600">Color = inherited ancestor state</span></div></div><div className="max-w-full rounded border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-medium text-blue-700">4 representative lanes; actual population has N particles</div></div>
        <div className="mb-4 grid min-w-0 gap-3 lg:grid-cols-[1fr_1.35fr]"><PromptCell active={step === 0} reducedMotion={reducedMotion} /><FormulaCard show={step >= 2} /></div>
        <div className="mb-2 hidden grid-cols-[1.15fr_120px_0.95fr] gap-3 px-3 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500 md:grid"><div>Particle trace / weight</div><div className="text-center">Ancestry map</div><div>KV state slice + tensors</div></div>
        <div className="grid min-w-0 max-w-full gap-2.5 overflow-hidden">{LANES.map((lane, index) => <LanePairRow key={lane.id} lane={lane} index={index} step={step} runtimeState={runtimeState} reducedMotion={reducedMotion} committedAncestry={committedAncestry} />)}</div>
        {step >= 2 && <motion.div className={`mt-4 grid place-items-center rounded border p-3 text-center ${statusInfo.tone}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}><div className="flex items-center gap-2 text-sm font-semibold"><AlertTriangle size={16} aria-hidden="true" /> {statusInfo.title}</div><div className="paper-mono mt-1 text-xs opacity-90">{statusInfo.detail}</div></motion.div>}
        {correspondenceActive && <FigureCaption label="Correspondence">The left and right P1–P4 lanes are the same visual particles. Resampling changes ancestry, so particle traces, KV state slices, and particle-bound tensors inherit through the same composed ancestor map.</FigureCaption>}
      </CardContent>
    </PaperCard>
  );
}

function CompactCacheComparison({ step, runtimeState, reducedMotion }) {
  const active = step >= 4 && runtimeState.needResample;
  const skipCarryForward = step >= 3 && !runtimeState.needResample;
  const cacheRows = useMemo(() => ANCESTOR_MAP.map((ancestor, newIndex) => ({ ancestor, newIndex, fromY: (ancestor - 1) * 30, toY: newIndex * 30 })), []);

  if (step < 3) {
    return (
      <PaperCard>
        <CardContent className="p-5">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-950"><Layers size={17} className="text-blue-700" aria-hidden="true" /> Cache detail inactive</div>
          <div className="rounded border border-slate-200 bg-slate-50 p-4 text-sm leading-6 text-slate-600">Cache reorder is only relevant after ESS has been checked and an ancestry-changing resampling event may occur. Before that point, the cached state is simply bound to each particle lane.</div>
        </CardContent>
      </PaperCard>
    );
  }

  return (
    <PaperCard>
      <CardContent className="p-5">
        <div className="mb-4 flex items-start justify-between gap-4"><div><div className="flex items-center gap-2 text-sm font-semibold text-slate-950"><Layers size={17} className="text-blue-700" aria-hidden="true" /> Cache-coherent reorder detail</div><div className="mt-1 text-xs leading-relaxed text-slate-500">The composed ancestor map is applied to every KV layer and particle-bound tensor.</div></div><span className="paper-mono rounded border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] text-slate-700">a = [3,1,3,4]</span></div>
        <div className="mb-3 flex flex-wrap gap-2 text-[11px]"><span className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600">KV layers</span><span className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600">position IDs</span><span className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600">attention masks</span><span className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600">decoding buffers</span></div>
        {skipCarryForward ? (
          <div className="rounded border border-emerald-200 bg-emerald-50 p-4 text-sm leading-6 text-emerald-800">No ancestry change → KV state is carried forward unchanged. Rebuild and gather are both avoided for this block.</div>
        ) : (
          <div className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]"><div className="rounded border border-red-200 bg-red-50 p-4"><div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-red-700">Naive rebuild</div>{["Drop cache", "Replay prefix", "Recompute attention"].map((label, index) => <motion.div key={label} className="mb-2 rounded border border-red-100 bg-white px-3 py-2 text-sm text-slate-700" animate={active && !reducedMotion ? { opacity: [0.55, 1, 0.55] } : { opacity: 0.55 }} transition={{ delay: index * 0.18, duration: 0.7, repeat: active && !reducedMotion ? Infinity : 0 }}>{index + 1}. {label}</motion.div>)}</div><div className="rounded border border-blue-200 bg-blue-50 p-4"><div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-blue-700">Gather cached state</div><div className="relative h-[132px] overflow-hidden rounded border border-blue-100 bg-white p-3"><AnimatePresence mode="wait">{!active ? <motion.div key="before" className="grid gap-1.5" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>{[1, 2, 3, 4].map((source) => <MiniCacheRow key={source} source={source} label={`Old P${source}`} />)}</motion.div> : <motion.div key="after" className="relative h-[112px]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>{cacheRows.map((row) => <motion.div key={row.newIndex} className="absolute left-0 top-0" initial={{ y: reducedMotion ? row.toY : row.fromY, opacity: 0.5 }} animate={{ y: row.toY, opacity: 1 }} transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}><MiniCacheRow source={row.ancestor} label={`New P${row.newIndex + 1} ← Old P${row.ancestor}`} /></motion.div>)}</motion.div>}</AnimatePresence></div></div></div>
        )}
        <FigureCaption label="Web Figure B">Cache-coherent resampling avoids prefix replay by gathering cached per-particle state and particle-bound tensors along the particle dimension.</FigureCaption>
      </CardContent>
    </PaperCard>
  );
}

function MiniCacheRow({ source, label }) {
  const color = CACHE_COLORS[source] ?? "#7f7f7f";
  return <div className="flex items-center gap-2"><div className="paper-mono w-28 text-right text-[11px] text-slate-600">{label}</div><div className="grid flex-1 grid-cols-5 gap-1">{CACHE_TOKENS.map((token) => <div key={token} className="h-5 rounded-sm border border-slate-200" style={{ background: `linear-gradient(135deg, ${color}dd, ${color}30)` }} />)}</div></div>;
}

function ResolvePanel({ step, runtimeState, reducedMotion }) {
  const active = step >= 5 || runtimeState.shouldTerminate || runtimeState.earlyExit;
  const title = runtimeState.shouldTerminate ? "Terminate at budget cap" : runtimeState.continueNextBlock ? "Continue next block" : runtimeState.shouldRestart ? "Fast-pass decision: restart" : runtimeState.earlyExit ? "Early vote" : "Weighted vote";
  const subtitle = runtimeState.shouldTerminate ? "Demo Cint exceeds the per-instance cap." : runtimeState.continueNextBlock ? "Resampled weights reset; decoding proceeds." : runtimeState.shouldRestart ? "Nfast → fresh Nhard pass from prompt." : runtimeState.earlyExit ? "Top-answer mass crosses δ during the Fast pass." : "Return top-answer particle or weighted vote.";
  const body = runtimeState.shouldTerminate ? "The paper enforces a strict per-instance compute cap; if the interaction budget is exhausted, inference terminates immediately. The budget bar here is a visual proxy." : runtimeState.continueNextBlock ? "After a resampling event, ASMC resets weights to 1/N and usually continues to the next token block rather than escalating immediately." : runtimeState.shouldRestart ? "At a guarded pass boundary, low confidence starts a fresh hard pass from the original prompt with a larger particle population." : runtimeState.earlyExit ? "The easy path demonstrates the paper's top-answer mass guard: if the population is already confident, ASMC can return the best particle early." : "The population is confident enough to produce an answer.";
  return <motion.div className={`rounded border p-5 ${active && (runtimeState.shouldRestart || runtimeState.shouldTerminate) ? "border-orange-200 bg-orange-50" : active ? "border-emerald-200 bg-emerald-50" : "border-slate-200 bg-white"}`} animate={active && !reducedMotion ? { scale: [1, 1.005, 1] } : { scale: 1 }}><div className="flex items-center gap-3"><div className={`grid h-10 w-10 place-items-center rounded text-white ${active && (runtimeState.shouldRestart || runtimeState.shouldTerminate) ? "bg-orange-500" : active ? "bg-emerald-600" : "bg-slate-300"}`}>{runtimeState.shouldRestart ? <Repeat2 size={20} aria-hidden="true" /> : <CheckCircle2 size={20} aria-hidden="true" />}</div><div><div className="text-base font-semibold text-slate-950">{title}</div><div className="text-sm text-slate-500">{subtitle}</div></div></div><motion.div className="mt-4 rounded border border-slate-200 bg-white p-4 text-sm leading-relaxed text-slate-600" animate={{ opacity: active ? 1 : 0.55 }}>{body}</motion.div></motion.div>;
}


export default function HeroWithModelAnimation({ onCopyBibtex, copyStatus }) {
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(DEFAULT_SPEED);
  const [nHard, setNHard] = useState(DEFAULT_N_HARD);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [mode, setMode] = useState(DEFAULT_DEMO_MODE);
  const [step, setStep] = useState(0);
  const [committedAncestry, setCommittedAncestry] = useState(null);
  const reducedMotion = useReducedMotion();
  const runtimeState = getRuntimeState(step, threshold, nHard, mode);
  const stopAtTerminalState = runtimeState.shouldTerminate || runtimeState.earlyExit || runtimeState.shouldOutput || runtimeState.shouldRestart;

  function advanceStep(currentStep) {
    const stateAtStep = getRuntimeState(currentStep, threshold, nHard, mode);
    if (stateAtStep.continueNextBlock && currentStep === STEPS.length - 1) {
      setCommittedAncestry(ANCESTOR_MAP);
      return 1;
    }
    return (currentStep + 1) % STEPS.length;
  }

  useEffect(() => {
    if (!isPlaying || stopAtTerminalState) return undefined;
    const timer = window.setInterval(() => setStep((prev) => advanceStep(prev)), speed);
    return () => window.clearInterval(timer);
  }, [isPlaying, speed, stopAtTerminalState, threshold, nHard, mode]);

  useEffect(() => {
    if (stopAtTerminalState) setIsPlaying(false);
  }, [stopAtTerminalState]);

  function resetAllControls() {
    setIsPlaying(false);
    setStep(0);
    setCommittedAncestry(null);
    setNHard(DEFAULT_N_HARD);
    setThreshold(DEFAULT_THRESHOLD);
    setSpeed(DEFAULT_SPEED);
    setMode(DEFAULT_DEMO_MODE);
  }

  function manualStepTo(nextStep) {
    setIsPlaying(false);
    if (nextStep === 0) setCommittedAncestry(null);
    setStep(nextStep);
  }

  function stepForward() {
    if (stopAtTerminalState) return;
    setIsPlaying(false);
    setStep((prev) => advanceStep(prev));
  }

  return (
    <section id="top" className="py-8 md:py-10">
      <div className="mx-auto max-w-5xl overflow-hidden px-1 text-center">
        <div className="mb-4 flex justify-center">
          <a
            href="https://icml.cc/virtual/2026/poster/64829"
            target="_blank"
            rel="noreferrer"
            aria-label="ICML 2026 poster page"
            className={`paper-mono inline-flex items-center rounded border border-slate-200 bg-white px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-slate-600 hover:border-slate-300 hover:bg-slate-50 hover:text-slate-950 ${focusClass()}`}
          >
            ICML 2026 <span className="mx-2 text-slate-300" aria-hidden="true">/</span> Poster 64829
          </a>
        </div>

        <h1 className="paper-serif text-[2.25rem] font-semibold leading-[1.02] tracking-tight text-slate-950 sm:text-5xl md:text-7xl">
          <span className="block sm:inline">Cache-Coherent</span>{" "}
          <span className="block sm:inline">ASMC</span>
        </h1>

        <p className="paper-pretty mx-auto mt-4 max-w-[21rem] text-lg leading-7 text-slate-700 sm:max-w-4xl md:text-xl md:leading-8">
          <span className="block sm:inline">Cache-Coherent Resampling</span>{" "}
          <span className="block sm:inline">for Efficient Test-Time Scaling</span>{" "}
          <span className="block sm:inline">in LLM Reasoning via</span>{" "}
          <span className="block sm:inline">Adaptive Sequential</span>{" "}
          <span className="block sm:inline">Monte Carlo</span>
        </p>

        <div className="mx-auto mt-3 flex max-w-3xl flex-wrap justify-center gap-x-2 gap-y-1 text-sm font-medium text-slate-700" aria-label="Authors">
          <span>Ke Wang</span>
          <span aria-hidden="true" className="text-slate-300">·</span>
          <span>Zehao Yu</span>
          <span aria-hidden="true" className="text-slate-300">·</span>
          <span>Luwei Wang</span>
          <span aria-hidden="true" className="text-slate-300">·</span>
          <span>Yongchao Huang</span>
        </div>

        <div className="mt-5 flex flex-wrap justify-center gap-2.5">
          <a
            href="https://openreview.net/pdf?id=JN6wxUGmW8"
            target="_blank"
            rel="noreferrer"
            className={`inline-flex items-center rounded bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-800 ${focusClass()}`}
          >
            <BookOpen className="mr-2" size={16} aria-hidden="true" /> Paper
          </a>
          <a
            href="https://github.com/Vicky-0256/ASMC"
            target="_blank"
            rel="noreferrer"
            className={`inline-flex items-center rounded border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 ${focusClass()}`}
          >
            <Github className="mr-2" size={16} aria-hidden="true" /> Code
          </a>
          <a
            href="#demo"
            className={`inline-flex items-center rounded border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 ${focusClass()}`}
          >
            Interactive Demo
          </a>
          <button
            type="button"
            onClick={onCopyBibtex}
            className={`inline-flex items-center rounded border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 ${focusClass()}`}
          >
            <Copy className="mr-2" size={16} aria-hidden="true" /> {copyStatus === "paper-bibtex" ? "Copied BibTeX" : "Copy BibTeX"}
          </button>
          <a
            href="https://github.com/Vicky-0256/ASMC/blob/main/docs/reproducibility.md"
            target="_blank"
            rel="noreferrer"
            className={`inline-flex items-center rounded border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 ${focusClass()}`}
          >
            Reproduce <ArrowRight className="ml-2" size={16} aria-hidden="true" />
          </a>
        </div>
        <div className="mt-2 flex flex-wrap justify-center gap-2">
          <a
            href="https://icml.cc/virtual/2026/poster/64829"
            target="_blank"
            rel="noreferrer"
            className={`inline-flex items-center rounded border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 hover:border-slate-300 hover:bg-slate-50 hover:text-slate-950 ${focusClass()}`}
          >
            Poster
          </a>
          <a
            href="#cite"
            className={`inline-flex items-center rounded border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 hover:border-slate-300 hover:bg-slate-50 hover:text-slate-950 ${focusClass()}`}
          >
            Cite this work
          </a>
        </div>
      </div>

      <div className="mt-7 grid min-w-0 gap-4 lg:grid-cols-[minmax(0,1fr)_312px] lg:items-start">
        <IntegratedMethodFigure step={step} runtimeState={runtimeState} threshold={threshold} reducedMotion={reducedMotion} committedAncestry={committedAncestry} />
        <HeroControlStrip
          isPlaying={isPlaying}
          setIsPlaying={setIsPlaying}
          resetAllControls={resetAllControls}
          stepForward={stepForward}
          nHard={nHard}
          setNHard={setNHard}
          threshold={threshold}
          setThreshold={setThreshold}
          speed={speed}
          setSpeed={setSpeed}
          mode={mode}
          setMode={setMode}
          step={step}
          setStep={manualStepTo}
          runtimeState={runtimeState}
          onManualStep={() => setIsPlaying(false)}
          terminalLocked={stopAtTerminalState}
          variant="side"
        />
      </div>

      <div className="mt-5 grid gap-4">
        <div className="grid gap-5 lg:grid-cols-[1fr_1.1fr]">
          <ResolvePanel step={step} runtimeState={runtimeState} reducedMotion={reducedMotion} />
          <CompactCacheComparison step={step} runtimeState={runtimeState} reducedMotion={reducedMotion} />
        </div>
        <FigureCaption label="Web Figure A">
          ASMC evolves particles by token blocks. The aligned rows show that P1–P4
          in the particle trace and P1–P4 in the KV state are the same representative
          visual lanes.
        </FigureCaption>
      </div>
    </section>
  );
}

"""
=============================================================================
Linguistic Agency and Emotional Framing in Medical Narratives
=============================================================================
A quantitative corpus-based pilot study examining how grammatical agency
(active vs. passive voice) correlates with emotional valence and affective
intensity in patient and caregiver medical narratives.

DEPENDENCIES — install before running:
    pip install spacy vaderSentiment pandas matplotlib scipy
    python -m spacy download en_core_web_sm

OUTPUT:
    /output/figures/     — 7 matplotlib figures (PNG)
    /output/tables/      — CSV tables (narrative & sentence features)
    /output/stats/       — CSV files for all statistical test results

CORPUS FORMAT (plain .txt files in /data/corpus/):
    ---NARRATIVE_START---
    id: P001
    author_type: patient
    source: Journal Name
    title: Narrative Title

    Full narrative text goes here in one or more paragraphs.
    ---NARRATIVE_END---

    author_type must be either "patient" or "caregiver".
=============================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import logging
from dataclasses import dataclass, field

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; change to "TkAgg" for pop-up windows
import matplotlib.pyplot as plt
from scipy import stats

# ── NLP ───────────────────────────────────────────────────────────────────────
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =============================================================================
# CONFIGURATION — edit paths here if needed
# =============================================================================
CORPUS_DIR  = "data/corpus"          # folder containing your .txt narrative files
OUTPUT_DIR  = "output"               # all outputs written here
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR  = os.path.join(OUTPUT_DIR, "tables")
STATS_DIR   = os.path.join(OUTPUT_DIR, "stats")
SPACY_MODEL = "en_core_web_sm"       # or "en_core_web_md" for better accuracy

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

for d in (FIGURES_DIR, TABLES_DIR, STATS_DIR):
    os.makedirs(d, exist_ok=True)


# =============================================================================
# SECTION 1 — CORPUS LOADING & PREPROCESSING
# =============================================================================

def load_corpus(corpus_dir: str) -> list[dict]:
    """
    Load all .txt narrative files from corpus_dir.
    Each file may contain multiple narratives delimited by
    ---NARRATIVE_START--- / ---NARRATIVE_END--- blocks.
    """
    narratives = []
    txt_files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{corpus_dir}'")

    for filename in txt_files:
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read()

        blocks = re.findall(
            r"---NARRATIVE_START---(.*?)---NARRATIVE_END---",
            content, re.DOTALL
        )
        for block in blocks:
            lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
            meta, body_lines, in_body = {}, [], False
            for line in lines:
                if not in_body and ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip().lower()] = v.strip()
                else:
                    in_body = True
                    body_lines.append(line)

            text = " ".join(body_lines)
            # Minimal normalisation: collapse whitespace, normalise quotes
            text = re.sub(r"\s+", " ", text).strip()
            text = text.replace("\u2018", "'").replace("\u2019", "'")

            narratives.append({
                "id":          meta.get("id", "UNKNOWN"),
                "author_type": meta.get("author_type", "unknown").lower(),
                "source":      meta.get("source", ""),
                "title":       meta.get("title", ""),
                "text":        text,
            })

    logger.info(f"Loaded {len(narratives)} narratives from {len(txt_files)} file(s).")
    return narratives


# =============================================================================
# SECTION 2 — AGENCY ANALYSIS (spaCy dependency parsing)
# =============================================================================

# Agency verbs common in medical self-advocacy discourse
AGENCY_VERBS = {
    "decide", "choose", "advocate", "insist", "request", "research",
    "manage", "coordinate", "push", "fight", "demand", "refuse",
    "challenge", "negotiate", "commit", "monitor", "build", "ask",
    "drive", "keep", "contact", "prepare", "document", "refuse",
    "escalate", "question", "track", "plan", "seek", "set",
}

@dataclass
class SentenceAgency:
    text:                   str
    is_passive:             bool
    is_agentless_passive:   bool
    has_first_person:       bool
    has_agency_verb:        bool
    agency_score:           float   # 0 = fully passive/agentless → 1 = fully active/agentic


@dataclass
class NarrativeAgency:
    narrative_id:           str
    author_type:            str
    total_sentences:        int
    passive_count:          int
    active_count:           int
    agentless_count:        int
    first_person_count:     int
    agency_verb_count:      int
    passive_ratio:          float
    active_ratio:           float
    mean_agency_score:      float
    sentences:              list[SentenceAgency] = field(default_factory=list)


def _parse_sentence_agency(sent) -> SentenceAgency:
    """Extract agency features from a single spaCy sentence span."""

    # ── Passive detection ─────────────────────────────────────────────────────
    # spaCy marks passive subjects with dep_ == "nsubjpass"
    # and passive auxiliaries with dep_ == "auxpass"
    passive_head_ids = set()
    has_by_agent = False

    for tok in sent:
        if tok.dep_ in {"nsubjpass", "auxpass"}:
            passive_head_ids.add(tok.head.i)
        if tok.dep_ == "agent" or (tok.lower_ == "by" and tok.dep_ == "prep"):
            has_by_agent = True

    is_passive   = len(passive_head_ids) > 0
    is_agentless = is_passive and not has_by_agent

    # ── First-person subject ──────────────────────────────────────────────────
    has_fp = any(
        tok.lower_ in {"i", "we"} and tok.dep_ in {"nsubj", "nsubjpass"}
        for tok in sent
    )

    # ── Agency verb ───────────────────────────────────────────────────────────
    has_av = any(tok.lemma_.lower() in AGENCY_VERBS for tok in sent)

    # ── Agency score (heuristic, range 0–1) ──────────────────────────────────
    # Weights chosen to reflect linguistic salience of each cue:
    #   +0.30 first-person subject  (strong volitional agent)
    #   +0.20 agency verb           (lexical marker of control)
    #   -0.30 passive voice         (grammatical reduction of agency)
    #   -0.20 agentless passive     (agent entirely absent from clause)
    score = 0.5
    if has_fp:   score += 0.30
    if has_av:   score += 0.20
    if is_passive:   score -= 0.30
    if is_agentless: score -= 0.20
    score = round(max(0.0, min(1.0, score)), 4)

    return SentenceAgency(
        text=sent.text.strip(),
        is_passive=is_passive,
        is_agentless_passive=is_agentless,
        has_first_person=has_fp,
        has_agency_verb=has_av,
        agency_score=score,
    )


def analyze_agency(nlp, narratives: list[dict]) -> list[NarrativeAgency]:
    """Run agency analysis across the full corpus."""
    results = []
    for narr in narratives:
        doc = nlp(narr["text"])
        sents = [
            _parse_sentence_agency(s)
            for s in doc.sents
            if len(s.text.split()) >= 3       # skip fragments
        ]
        n = len(sents)
        passive  = sum(1 for s in sents if s.is_passive)
        active   = sum(1 for s in sents if not s.is_passive)
        agentless = sum(1 for s in sents if s.is_agentless_passive)
        fp_count  = sum(1 for s in sents if s.has_first_person)
        av_count  = sum(1 for s in sents if s.has_agency_verb)

        results.append(NarrativeAgency(
            narrative_id=narr["id"],
            author_type=narr["author_type"],
            total_sentences=n,
            passive_count=passive,
            active_count=active,
            agentless_count=agentless,
            first_person_count=fp_count,
            agency_verb_count=av_count,
            passive_ratio=round(passive / n, 4) if n else 0.0,
            active_ratio=round(active  / n, 4) if n else 0.0,
            mean_agency_score=round(
                sum(s.agency_score for s in sents) / n, 4
            ) if n else 0.0,
            sentences=sents,
        ))
        logger.info(f"Agency analysed: {narr['id']} — {n} sentences, "
                    f"passive_ratio={results[-1].passive_ratio:.2f}")
    return results


# =============================================================================
# SECTION 3 — SENTIMENT ANALYSIS (VADER)
# =============================================================================

# Domain-specific affective lexicon for medical narratives
AFFECTIVE_LEXICON = {
    # Negative affect
    "pain", "suffering", "helpless", "invisible", "powerless", "fear",
    "anxious", "desperate", "grief", "loss", "dread", "overwhelmed",
    "defeat", "exhausted", "mourned", "eroded", "deteriorated", "burden",
    # Positive affect / resilience
    "hope", "strength", "courage", "determined", "empowered", "advocate",
    "resilient", "committed", "grateful", "proud", "breakthrough",
    # Intensity amplifiers
    "never", "always", "constantly", "deeply", "profoundly", "every",
}


@dataclass
class SentenceSentiment:
    text:               str
    compound:           float   # VADER compound: -1 (most negative) → +1 (most positive)
    positive:           float
    negative:           float
    neutral:            float
    label:              str     # "positive" | "negative" | "neutral"
    intensity:          float   # |compound| — emotional strength regardless of polarity
    affective_words:    int     # count of domain affective lexicon matches


@dataclass
class NarrativeSentiment:
    narrative_id:               str
    author_type:                str
    mean_compound:              float
    mean_intensity:             float
    mean_positive:              float
    mean_negative:              float
    mean_neutral:               float
    overall_label:              str
    negative_sentence_ratio:    float
    positive_sentence_ratio:    float
    total_affective_words:      int
    affective_density:          float   # affective words per sentence
    sentences:                  list[SentenceSentiment] = field(default_factory=list)


def _label(compound: float) -> str:
    if compound >=  0.05: return "positive"
    if compound <= -0.05: return "negative"
    return "neutral"


def analyze_sentiment(sia, narratives: list[dict], agency_results: list[NarrativeAgency]) -> list[NarrativeSentiment]:
    """
    Run VADER sentiment on each sentence.
    Uses sentence boundaries from the already-parsed agency results
    to guarantee consistent segmentation across both modules.
    """
    agency_map = {ar.narrative_id: ar.sentences for ar in agency_results}
    results = []

    for narr in narratives:
        sent_texts = [s.text for s in agency_map.get(narr["id"], [])]
        if not sent_texts:
            # Fallback: naive sentence split if agency data missing
            sent_texts = [s for s in re.split(r"(?<=[.!?])\s+", narr["text"]) if len(s.split()) >= 3]

        sents = []
        for txt in sent_texts:
            sc = sia.polarity_scores(txt)
            compound = round(sc["compound"], 4)
            aff = len(set(txt.lower().split()) & AFFECTIVE_LEXICON)
            sents.append(SentenceSentiment(
                text=txt,
                compound=compound,
                positive=round(sc["pos"], 4),
                negative=round(sc["neg"], 4),
                neutral=round(sc["neu"], 4),
                label=_label(compound),
                intensity=round(abs(compound), 4),
                affective_words=aff,
            ))

        n = len(sents)
        mean_cpd   = round(sum(s.compound  for s in sents) / n, 4) if n else 0.0
        mean_int   = round(sum(s.intensity for s in sents) / n, 4) if n else 0.0
        mean_pos   = round(sum(s.positive  for s in sents) / n, 4) if n else 0.0
        mean_neg   = round(sum(s.negative  for s in sents) / n, 4) if n else 0.0
        mean_neu   = round(sum(s.neutral   for s in sents) / n, 4) if n else 0.0
        total_aff  = sum(s.affective_words for s in sents)

        results.append(NarrativeSentiment(
            narrative_id=narr["id"],
            author_type=narr["author_type"],
            mean_compound=mean_cpd,
            mean_intensity=mean_int,
            mean_positive=mean_pos,
            mean_negative=mean_neg,
            mean_neutral=mean_neu,
            overall_label=_label(mean_cpd),
            negative_sentence_ratio=round(sum(1 for s in sents if s.label == "negative") / n, 4) if n else 0.0,
            positive_sentence_ratio=round(sum(1 for s in sents if s.label == "positive") / n, 4) if n else 0.0,
            total_affective_words=total_aff,
            affective_density=round(total_aff / n, 4) if n else 0.0,
            sentences=sents,
        ))
        logger.info(f"Sentiment analysed: {narr['id']} — mean_compound={results[-1].mean_compound:.3f}")
    return results


# =============================================================================
# SECTION 4 — FEATURE EXTRACTION → pandas DataFrames
# =============================================================================

def build_narrative_df(agency_results, sentiment_results) -> pd.DataFrame:
    """One row per narrative; merges agency + sentiment aggregate features."""
    sent_map = {s.narrative_id: s for s in sentiment_results}
    rows = []
    for ag in agency_results:
        sn = sent_map.get(ag.narrative_id)
        if not sn:
            continue
        rows.append({
            "narrative_id":             ag.narrative_id,
            "author_type":              ag.author_type,
            # Agency
            "total_sentences":          ag.total_sentences,
            "passive_count":            ag.passive_count,
            "active_count":             ag.active_count,
            "agentless_count":          ag.agentless_count,
            "first_person_count":       ag.first_person_count,
            "agency_verb_count":        ag.agency_verb_count,
            "passive_ratio":            ag.passive_ratio,
            "active_ratio":             ag.active_ratio,
            "mean_agency_score":        ag.mean_agency_score,
            # Sentiment
            "mean_compound":            sn.mean_compound,
            "mean_intensity":           sn.mean_intensity,
            "mean_positive":            sn.mean_positive,
            "mean_negative":            sn.mean_negative,
            "mean_neutral":             sn.mean_neutral,
            "overall_sentiment":        sn.overall_label,
            "negative_sentence_ratio":  sn.negative_sentence_ratio,
            "positive_sentence_ratio":  sn.positive_sentence_ratio,
            "total_affective_words":    sn.total_affective_words,
            "affective_density":        sn.affective_density,
        })
    return pd.DataFrame(rows)


def build_sentence_df(agency_results, sentiment_results) -> pd.DataFrame:
    """One row per sentence; aligns agency + sentiment sentence-level features."""
    sent_map = {s.narrative_id: s for s in sentiment_results}
    rows = []
    for ag in agency_results:
        sn = sent_map.get(ag.narrative_id)
        if not sn:
            continue
        for i, (ag_s, sn_s) in enumerate(
            zip(ag.sentences, sn.sentences)
        ):
            rows.append({
                "narrative_id":         ag.narrative_id,
                "author_type":          ag.author_type,
                "sentence_index":       i,
                "sentence_text":        ag_s.text,
                # Agency
                "is_passive":           ag_s.is_passive,
                "is_agentless":         ag_s.is_agentless_passive,
                "has_first_person":     ag_s.has_first_person,
                "has_agency_verb":      ag_s.has_agency_verb,
                "agency_score":         ag_s.agency_score,
                "voice":                "passive" if ag_s.is_passive else "active",
                # Sentiment
                "compound":             sn_s.compound,
                "positive":             sn_s.positive,
                "negative":             sn_s.negative,
                "neutral":              sn_s.neutral,
                "sentiment_label":      sn_s.label,
                "intensity":            sn_s.intensity,
                "affective_words":      sn_s.affective_words,
            })
    return pd.DataFrame(rows)


def compute_frequency_distributions(sentence_df: pd.DataFrame) -> dict:
    """Key frequency tables for reporting and plotting."""
    return {
        "agency_by_author": (
            sentence_df.groupby(["author_type", "voice"]).size().reset_index(name="count")
        ),
        "sentiment_by_voice": (
            sentence_df.groupby(["voice", "sentiment_label"]).size().reset_index(name="count")
        ),
        "means_by_author": (
            sentence_df.groupby("author_type")[
                ["agency_score", "compound", "intensity", "affective_words"]
            ].mean().round(4)
        ),
        "means_by_voice": (
            sentence_df.groupby("voice")[
                ["compound", "intensity", "negative", "positive", "affective_words"]
            ].mean().round(4)
        ),
        "means_by_first_person": (
            sentence_df.groupby("has_first_person")[
                ["compound", "intensity", "negative"]
            ].mean().round(4)
        ),
    }


# =============================================================================
# SECTION 5 — STATISTICAL ANALYSIS
# =============================================================================

def descriptive_stats(df: pd.DataFrame, group_col: str, metric_cols: list[str]) -> pd.DataFrame:
    """Mean, median, SD, IQR per group × metric."""
    rows = []
    for gval, gdf in df.groupby(group_col):
        for col in metric_cols:
            s = gdf[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            rows.append({
                "group": gval, "metric": col, "n": len(s),
                "mean": round(s.mean(), 4), "median": round(s.median(), 4),
                "SD": round(s.std(), 4), "IQR": round(q3 - q1, 4),
                "min": round(s.min(), 4), "max": round(s.max(), 4),
            })
    return pd.DataFrame(rows)


def mann_whitney(df, group_col, group_a, group_b, metrics) -> pd.DataFrame:
    """
    Mann-Whitney U test (two-sided, non-parametric) comparing two groups.
    Appropriate for small samples without normality assumption.
    Effect size r = Z / sqrt(N).
    """
    rows = []
    a_df = df[df[group_col] == group_a]
    b_df = df[df[group_col] == group_b]
    for col in metrics:
        a = a_df[col].dropna().values
        b = b_df[col].dropna().values
        if len(a) < 2 or len(b) < 2:
            rows.append({"metric": col, "U": None, "p_value": None,
                         "effect_r": None, "sig_p05": None,
                         f"mean_{group_a}": round(a.mean(), 4) if len(a) else None,
                         f"mean_{group_b}": round(b.mean(), 4) if len(b) else None})
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        z = stats.norm.isf(p / 2)
        r = round(z / ((len(a) + len(b)) ** 0.5), 4)
        rows.append({
            "metric": col,
            f"mean_{group_a}": round(a.mean(), 4),
            f"mean_{group_b}": round(b.mean(), 4),
            "U": round(u, 2), "p_value": round(p, 4),
            "effect_r": r, "sig_p05": p < 0.05,
        })
    return pd.DataFrame(rows)


def spearman_corr(df, predictor, outcomes) -> pd.DataFrame:
    """Spearman ρ between predictor and each outcome variable."""
    rows = []
    x = df[predictor].dropna()
    for outcome in outcomes:
        y = df[outcome].dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < 4:
            rows.append({"predictor": predictor, "outcome": outcome,
                         "rho": None, "p_value": None, "n": len(idx), "sig_p05": None})
            continue
        rho, p = stats.spearmanr(x.loc[idx], y.loc[idx])
        rows.append({
            "predictor": predictor, "outcome": outcome,
            "rho": round(rho, 4), "p_value": round(p, 4),
            "n": len(idx), "sig_p05": p < 0.05,
        })
    return pd.DataFrame(rows)


def run_statistics(narrative_df: pd.DataFrame, sentence_df: pd.DataFrame) -> dict:
    """Run the full battery of statistical tests and return results dict."""
    sent_metrics  = ["compound", "intensity", "negative", "positive", "affective_words"]
    narr_sent_m   = ["mean_compound", "mean_intensity", "mean_negative",
                     "mean_positive", "affective_density"]
    narr_agency_m = ["mean_agency_score", "passive_ratio", "active_ratio",
                     "first_person_count", "agency_verb_count"]

    return {
        # Descriptive
        "desc_by_voice":   descriptive_stats(sentence_df,  "voice",       sent_metrics + ["agency_score"]),
        "desc_by_author":  descriptive_stats(narrative_df, "author_type", narr_agency_m + narr_sent_m),
        # Mann-Whitney comparisons
        "mw_voice":        mann_whitney(sentence_df,  "voice",       "active",   "passive",    sent_metrics),
        "mw_author_sent":  mann_whitney(narrative_df, "author_type", "patient",  "caregiver",  narr_sent_m),
        "mw_author_agency":mann_whitney(narrative_df, "author_type", "patient",  "caregiver",  narr_agency_m),
        # Spearman correlations
        "spearman_sent":   spearman_corr(sentence_df,  "agency_score",      sent_metrics),
        "spearman_narr":   spearman_corr(narrative_df, "mean_agency_score", narr_sent_m),
    }


# =============================================================================
# SECTION 6 — VISUALISATION
# =============================================================================

PALETTE = {
    "patient":   "#3A7CA5",
    "caregiver": "#E07B39",
    "active":    "#2E8B57",
    "passive":   "#C0392B",
    "neutral":   "#7F8C8D",
}
plt.rc("font", family="serif", size=11)
plt.rc("axes", facecolor="#FAFAFA", edgecolor="#CCCCCC",
       titleweight="bold", titlesize=12)
plt.rc("figure", facecolor="white")
plt.rc("grid", color="#E8E8E8", linestyle="--", linewidth=0.7)


def _save(fig, name: str) -> str:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def fig1_agency_by_author(sentence_df: pd.DataFrame):
    """Stacked bar — active vs passive sentence counts by author type."""
    counts = (sentence_df.groupby(["author_type", "voice"])
              .size().unstack(fill_value=0))
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for group, color in [("active", PALETTE["active"]), ("passive", PALETTE["passive"])]:
        if group not in counts.columns:
            continue
        vals = counts[group].values
        ax.bar(x, vals, 0.5, bottom=bottom, color=color,
               label=group.capitalize(), alpha=0.88, edgecolor="white")
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in counts.index])
    ax.set_ylabel("Sentence count")
    ax.set_title("Fig. 1 — Active vs Passive Constructions by Author Type")
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    return _save(fig, "fig1_agency_by_author.png")


def fig2_compound_by_voice(sentence_df: pd.DataFrame):
    """Box plots — VADER compound score for active vs passive sentences."""
    groups = ["active", "passive"]
    data   = [sentence_df[sentence_df["voice"] == g]["compound"].dropna().values for g in groups]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(PALETTE[g]); patch.set_alpha(0.75)
    np.random.seed(42)
    for i, (d, g) in enumerate(zip(data, groups), 1):
        ax.scatter(i + np.random.uniform(-0.12, 0.12, len(d)), d,
                   alpha=0.55, s=22, color=PALETTE[g], zorder=3)
    ax.axhline(0, color=PALETTE["neutral"], linestyle=":", linewidth=1.2, label="Neutral (0)")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Active voice", "Passive voice"])
    ax.set_ylabel("VADER compound score")
    ax.set_title("Fig. 2 — Sentiment Compound Score by Voice Construction")
    ax.yaxis.grid(True); ax.set_axisbelow(True); ax.legend(fontsize=9)
    return _save(fig, "fig2_compound_by_voice.png")


def fig3_scatter(sentence_df: pd.DataFrame):
    """Scatter — agency score vs VADER compound, coloured by author type."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for author in ["patient", "caregiver"]:
        sub = sentence_df[sentence_df["author_type"] == author]
        ax.scatter(sub["agency_score"], sub["compound"],
                   c=PALETTE[author], label=author.capitalize(),
                   alpha=0.65, s=50, edgecolors="white", linewidths=0.5)
    x = sentence_df["agency_score"].dropna()
    y = sentence_df["compound"].dropna()
    idx = x.index.intersection(y.index)
    if len(idx) >= 3:
        m, b = np.polyfit(x.loc[idx], y.loc[idx], 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color=PALETTE["neutral"], linestyle="--",
                linewidth=1.4, label="Linear trend")
    ax.axhline(0, color="#CCCCCC", linestyle=":", linewidth=1)
    ax.axvline(0.5, color="#CCCCCC", linestyle=":", linewidth=1)
    ax.set_xlabel("Agency score  (0 = passive/agentless → 1 = active/agentic)")
    ax.set_ylabel("VADER compound sentiment")
    ax.set_title("Fig. 3 — Agency Score vs Sentiment Compound (sentence level)")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.yaxis.grid(True); ax.xaxis.grid(True); ax.set_axisbelow(True)
    return _save(fig, "fig3_agency_sentiment_scatter.png")


def fig4_negative_by_voice(sentence_df: pd.DataFrame):
    """Bar chart — mean VADER negative score for active vs passive sentences."""
    means  = sentence_df.groupby("voice")["negative"].mean()
    errors = sentence_df.groupby("voice")["negative"].sem()
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [PALETTE.get(g, PALETTE["neutral"]) for g in means.index]
    bars = ax.bar([g.capitalize() for g in means.index], means.values,
                  yerr=errors.values, color=colors, alpha=0.82,
                  edgecolor="white", capsize=6, width=0.5)
    for bar, v in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean VADER negative score (± SE)")
    ax.set_title("Fig. 4 — Negative Affect Ratio by Voice Construction")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    return _save(fig, "fig4_negative_by_voice.png")


def fig5_affective_density(sentence_df: pd.DataFrame):
    """Grouped bar — affective word density by author type and voice."""
    pivot = (sentence_df.groupby(["author_type", "voice"])["affective_words"]
             .mean().unstack(fill_value=0))
    x = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (group, color) in enumerate([("active", PALETTE["active"]),
                                         ("passive", PALETTE["passive"])]):
        if group not in pivot.columns:
            continue
        offset = (i - 0.5) * 0.35
        bars = ax.bar(x + offset, pivot[group].values, 0.35,
                      label=group.capitalize(), color=color, alpha=0.82, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in pivot.index])
    ax.set_ylabel("Mean affective words per sentence")
    ax.set_title("Fig. 5 — Affective Lexicon Density by Author Type & Voice")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    return _save(fig, "fig5_affective_density.png")


def fig6_heatmap(narrative_df: pd.DataFrame):
    """Spearman correlation heatmap of narrative-level agency and sentiment metrics."""
    cols = ["mean_agency_score", "passive_ratio", "first_person_count",
            "mean_compound", "mean_intensity", "mean_negative", "affective_density"]
    labels = ["Agency\nscore", "Passive\nratio", "1st-person\ncount",
              "Compound\nsentiment", "Emotional\nintensity", "Negative\naffect",
              "Affective\ndensity"]
    avail = [c for c in cols if c in narrative_df.columns]
    avail_labels = [labels[cols.index(c)] for c in avail]
    corr = narrative_df[avail].corr(method="spearman").values
    n = len(avail)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(avail_labels, fontsize=9)
    ax.set_yticklabels(avail_labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(v) > 0.6 else "black")
    ax.set_title("Fig. 6 — Spearman Correlations: Agency & Sentiment (narrative level)")
    fig.tight_layout()
    return _save(fig, "fig6_correlation_heatmap.png")


def fig7_timelines(sentence_df: pd.DataFrame):
    """Dual-axis line chart per narrative — agency score & compound over sentence index."""
    paths = []
    for nid in sentence_df["narrative_id"].unique():
        sub    = sentence_df[sentence_df["narrative_id"] == nid].reset_index(drop=True)
        author = sub["author_type"].iloc[0]
        color  = PALETTE.get(author, PALETTE["neutral"])
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(sub.index, sub["agency_score"], color=color, marker="o",
                 linewidth=1.8, markersize=5, label="Agency score")
        ax1.set_ylabel("Agency score", color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(-0.05, 1.1)
        ax2 = ax1.twinx()
        ax2.plot(sub.index, sub["compound"], color=PALETTE["neutral"], marker="s",
                 linewidth=1.4, markersize=4, linestyle="--", label="Sentiment compound")
        ax2.axhline(0, color="#CCCCCC", linestyle=":", linewidth=1)
        ax2.set_ylabel("Sentiment compound", color=PALETTE["neutral"])
        ax2.tick_params(axis="y", labelcolor=PALETTE["neutral"])
        ax2.set_ylim(-1.1, 1.1)
        ax1.set_xlabel("Sentence index")
        ax1.set_title(f"Fig. 7 — {nid} ({author.capitalize()}): Agency & Sentiment Timeline")
        ax1.xaxis.grid(True); ax1.set_axisbelow(True)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
        paths.append(_save(fig, f"fig7_timeline_{nid}.png"))
    return paths


def generate_all_figures(narrative_df, sentence_df):
    logger.info("Generating figures...")
    fig1_agency_by_author(sentence_df)
    fig2_compound_by_voice(sentence_df)
    fig3_scatter(sentence_df)
    fig4_negative_by_voice(sentence_df)
    fig5_affective_density(sentence_df)
    fig6_heatmap(narrative_df)
    fig7_timelines(sentence_df)
    logger.info(f"All figures saved to {FIGURES_DIR}")


# =============================================================================
# SECTION 7 — SAVE OUTPUTS
# =============================================================================

def save_all(narrative_df, sentence_df, freq_dists, stats_results):
    """Write all CSV outputs to /output/tables/ and /output/stats/."""

    # DataFrames
    narrative_df.to_csv(os.path.join(TABLES_DIR, "narrative_features.csv"), index=False)
    sentence_df.to_csv( os.path.join(TABLES_DIR, "sentence_features.csv"),  index=False)
    for name, df in freq_dists.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(os.path.join(TABLES_DIR, f"freq_{name}.csv"), index=True)

    # Statistical results
    for name, df in stats_results.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(os.path.join(STATS_DIR, f"stats_{name}.csv"), index=False)

    logger.info(f"Tables saved to {TABLES_DIR}")
    logger.info(f"Stats saved to  {STATS_DIR}")


# =============================================================================
# SECTION 8 — CONSOLE REPORT
# =============================================================================

def print_report(narrative_df, sentence_df, freq_dists, stats_results):
    """Print a readable summary of key quantitative findings to the console."""

    sep = "=" * 70
    print(f"\n{sep}")
    print("  LINGUISTIC AGENCY AND EMOTIONAL FRAMING IN MEDICAL NARRATIVES")
    print(f"  Quantitative Pilot Study — Summary Report")
    print(sep)

    print(f"\nCORPUS OVERVIEW")
    print(f"  Total narratives : {len(narrative_df)}")
    for atype, g in narrative_df.groupby("author_type"):
        print(f"    {atype.capitalize():12s}: {len(g)} narratives, "
              f"{int(g['total_sentences'].sum())} sentences")

    print(f"\nFREQUENCY DISTRIBUTIONS")
    print("  Agency constructions by author type:")
    print(freq_dists["agency_by_author"].to_string(index=False))
    print("\n  Mean metrics by voice (active / passive) — sentence level:")
    print(freq_dists["means_by_voice"].to_string())
    print("\n  Mean metrics by author type — sentence level:")
    print(freq_dists["means_by_author"].to_string())

    print(f"\nMANN-WHITNEY U TESTS")
    for key, label in [
        ("mw_voice",         "Active vs Passive → Sentiment (sentence level)"),
        ("mw_author_agency", "Patient vs Caregiver → Agency (narrative level)"),
        ("mw_author_sent",   "Patient vs Caregiver → Sentiment (narrative level)"),
    ]:
        df = stats_results.get(key)
        if df is not None and not df.empty:
            print(f"\n  {label}:")
            print(df.to_string(index=False))

    print(f"\nSPEARMAN CORRELATIONS")
    for key, label in [
        ("spearman_sent", "Agency Score → Sentiment (sentence level)"),
        ("spearman_narr", "Mean Agency Score → Sentiment (narrative level)"),
    ]:
        df = stats_results.get(key)
        if df is not None and not df.empty:
            print(f"\n  {label}:")
            print(df.to_string(index=False))

    print(f"\n{sep}")
    print("  OUTPUT FILES")
    print(f"  Figures  : {FIGURES_DIR}")
    print(f"  Tables   : {TABLES_DIR}")
    print(f"  Stats    : {STATS_DIR}")
    print(sep + "\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    logger.info("Loading spaCy model...")
    nlp = spacy.load(SPACY_MODEL)

    logger.info("Initialising VADER...")
    sia = SentimentIntensityAnalyzer()

    # ── 1. Load corpus ────────────────────────────────────────────────────────
    narratives = load_corpus(CORPUS_DIR)

    # ── 2. Agency analysis ────────────────────────────────────────────────────
    logger.info("Running agency analysis...")
    agency_results = analyze_agency(nlp, narratives)

    # ── 3. Sentiment analysis ─────────────────────────────────────────────────
    logger.info("Running sentiment analysis...")
    sentiment_results = analyze_sentiment(sia, narratives, agency_results)

    # ── 4. Build DataFrames ───────────────────────────────────────────────────
    logger.info("Building feature DataFrames...")
    narrative_df = build_narrative_df(agency_results, sentiment_results)
    sentence_df  = build_sentence_df(agency_results, sentiment_results)

    # ── 5. Frequency distributions ────────────────────────────────────────────
    freq_dists = compute_frequency_distributions(sentence_df)

    # ── 6. Statistical analysis ───────────────────────────────────────────────
    logger.info("Running statistical tests...")
    stats_results = run_statistics(narrative_df, sentence_df)

    # ── 7. Visualisation ──────────────────────────────────────────────────────
    generate_all_figures(narrative_df, sentence_df)

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    save_all(narrative_df, sentence_df, freq_dists, stats_results)

    # ── 9. Console report ─────────────────────────────────────────────────────
    print_report(narrative_df, sentence_df, freq_dists, stats_results)


if __name__ == "__main__":
    main()

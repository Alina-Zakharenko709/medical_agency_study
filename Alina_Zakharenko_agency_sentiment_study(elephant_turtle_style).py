"""
Linguistic Agency and Emotional Framing in Medical Narratives
-------------------------------------------------------------
BEFORE RUNNING, install libraries once in your terminal:
    pip install spacy vaderSentiment pandas matplotlib scipy
    python -m spacy download en_core_web_sm

Then run with:
    python agency_sentiment_study_simple.py

Results appear in the output/ folder.
"""

# These are the libraries we need — like importing tools from a toolbox
import os
import re
import spacy
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves figures as files instead of opening pop-up windows
import matplotlib.pyplot as plt
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =============================================================================
# SETTINGS — change these if your folders are named differently
# =============================================================================
CORPUS_FOLDER  = "data/corpus"       # where your .txt narrative files live
OUTPUT_FOLDER  = "output"            # where results will be saved
FIGURES_FOLDER = "output/figures"
TABLES_FOLDER  = "output/tables"
STATS_FOLDER   = "output/stats"

# Create output folders if they don't exist yet
for folder in [FIGURES_FOLDER, TABLES_FOLDER, STATS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Words that signal a person is actively in control (medical context)
AGENCY_VERBS = {
    "decide", "choose", "advocate", "insist", "request", "research",
    "manage", "coordinate", "push", "fight", "demand", "refuse",
    "challenge", "negotiate", "commit", "monitor", "ask", "contact",
    "prepare", "document", "escalate", "question", "track", "plan",
    "seek", "set", "drive", "keep", "build",
}

# Words that carry strong emotion in medical narratives
AFFECTIVE_WORDS = {
    "pain", "suffering", "helpless", "invisible", "powerless", "fear",
    "anxious", "desperate", "grief", "loss", "dread", "overwhelmed",
    "defeat", "exhausted", "mourned", "deteriorated", "burden",
    "hope", "strength", "courage", "determined", "empowered",
    "resilient", "committed", "grateful", "proud", "breakthrough",
    "never", "always", "constantly", "deeply", "profoundly",
}


# =============================================================================
# STEP 1 — LOAD THE CORPUS
# Read all .txt files and extract each narrative as a dictionary
# =============================================================================

def load_corpus():
    """Read narrative files and return a list of narratives."""
    narratives = []

    # Get all .txt files in the corpus folder
    files = sorted(f for f in os.listdir(CORPUS_FOLDER) if f.endswith(".txt"))
    if not files:
        print("ERROR: No .txt files found in", CORPUS_FOLDER)
        return narratives

    for filename in files:
        filepath = os.path.join(CORPUS_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Find each block between ---NARRATIVE_START--- and ---NARRATIVE_END---
        blocks = re.findall(
            r"---NARRATIVE_START---(.*?)---NARRATIVE_END---",
            content, re.DOTALL
        )

        for block in blocks:
            lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
            metadata = {}
            text_lines = []
            reading_text = False

            for line in lines:
                # Lines with ":" before the text body are metadata (id, author_type, etc.)
                if not reading_text and ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip().lower()] = value.strip()
                else:
                    reading_text = True
                    text_lines.append(line)

            # Clean up the text
            text = " ".join(text_lines)
            text = re.sub(r"\s+", " ", text).strip()

            narratives.append({
                "id":          metadata.get("id", "UNKNOWN"),
                "author_type": metadata.get("author_type", "unknown").lower(),
                "title":       metadata.get("title", ""),
                "text":        text,
            })

    print(f"Loaded {len(narratives)} narratives from {len(files)} file(s).")
    return narratives


# =============================================================================
# STEP 2 — AGENCY ANALYSIS
# For each sentence, detect: passive voice, first-person subject, agency verbs
# Then compute an agency score between 0 (very passive) and 1 (very active)
# =============================================================================

def analyze_sentence_agency(sentence):
    """
    Takes one spaCy sentence and returns a dictionary of agency features.
    spaCy parses the grammar so we can detect passive constructions.
    """
    # Detect passive voice using spaCy's dependency labels
    # "nsubjpass" = passive subject, "auxpass" = passive auxiliary (e.g. "was told")
    passive_heads = set()
    has_by_phrase = False

    for token in sentence:
        if token.dep_ in {"nsubjpass", "auxpass"}:
            passive_heads.add(token.head.i)
        if token.lower_ == "by" and token.dep_ == "prep":
            has_by_phrase = True

    is_passive   = len(passive_heads) > 0
    is_agentless = is_passive and not has_by_phrase  # passive with NO "by ..." phrase

    # Check if the subject is "I" or "we" (first-person = strong agency signal)
    has_first_person = any(
        token.lower_ in {"i", "we"} and token.dep_ in {"nsubj", "nsubjpass"}
        for token in sentence
    )

    # Check if the sentence contains any of our agency verbs
    has_agency_verb = any(token.lemma_.lower() in AGENCY_VERBS for token in sentence)

    # Compute agency score (starts at neutral 0.5, then adjusts)
    score = 0.5
    if has_first_person: score += 0.30   # "I decided" = high agency
    if has_agency_verb:  score += 0.20   # "I managed" = high agency
    if is_passive:       score -= 0.30   # "was told" = reduced agency
    if is_agentless:     score -= 0.20   # "was told" (no "by whom") = even less agency
    score = round(max(0.0, min(1.0, score)), 4)  # keep within 0–1

    return {
        "is_passive":        is_passive,
        "is_agentless":      is_agentless,
        "has_first_person":  has_first_person,
        "has_agency_verb":   has_agency_verb,
        "agency_score":      score,
        "voice":             "passive" if is_passive else "active",
    }


def analyze_agency(nlp, narratives):
    """
    Run agency analysis on every narrative.
    Returns a list of result dictionaries (one per narrative).
    """
    results = []

    for narr in narratives:
        doc = nlp(narr["text"])  # spaCy parses the whole text

        # Analyse each sentence (skip very short ones)
        sentences = []
        for sent in doc.sents:
            if len(sent.text.split()) >= 3:
                agency = analyze_sentence_agency(sent)
                agency["text"] = sent.text.strip()
                sentences.append(agency)

        n = len(sentences)
        passive_count = sum(1 for s in sentences if s["is_passive"])
        active_count  = sum(1 for s in sentences if not s["is_passive"])

        results.append({
            "narrative_id":      narr["id"],
            "author_type":       narr["author_type"],
            "total_sentences":   n,
            "passive_count":     passive_count,
            "active_count":      active_count,
            "agentless_count":   sum(1 for s in sentences if s["is_agentless"]),
            "first_person_count":sum(1 for s in sentences if s["has_first_person"]),
            "agency_verb_count": sum(1 for s in sentences if s["has_agency_verb"]),
            "passive_ratio":     round(passive_count / n, 4) if n else 0,
            "active_ratio":      round(active_count  / n, 4) if n else 0,
            "mean_agency_score": round(sum(s["agency_score"] for s in sentences) / n, 4) if n else 0,
            "sentences":         sentences,  # keep sentence-level data for later
        })

        print(f"  Agency: {narr['id']} — {n} sentences, passive_ratio={results[-1]['passive_ratio']:.2f}")

    return results


# =============================================================================
# STEP 3 — SENTIMENT ANALYSIS
# Use VADER to score each sentence for positive/negative/neutral sentiment
# =============================================================================

def analyze_sentiment(sia, narratives, agency_results):
    """
    Run VADER sentiment on every sentence.
    Uses the same sentence boundaries as the agency analysis.
    """
    # Build a quick lookup: narrative id → its sentences (from agency step)
    agency_map = {r["narrative_id"]: r["sentences"] for r in agency_results}
    results = []

    for narr in narratives:
        sentence_texts = [s["text"] for s in agency_map.get(narr["id"], [])]

        sentences = []
        for text in sentence_texts:
            scores = sia.polarity_scores(text)  # VADER returns pos, neg, neu, compound
            compound = round(scores["compound"], 4)

            # Count how many affective words appear in this sentence
            words_in_sentence = set(text.lower().split())
            affective_count = len(words_in_sentence & AFFECTIVE_WORDS)

            # Label the sentence sentiment
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"

            sentences.append({
                "text":            text,
                "compound":        compound,
                "positive":        round(scores["pos"], 4),
                "negative":        round(scores["neg"], 4),
                "neutral":         round(scores["neu"], 4),
                "label":           label,
                "intensity":       round(abs(compound), 4),  # strength, ignoring polarity
                "affective_words": affective_count,
            })

        n = len(sentences)
        mean_compound = round(sum(s["compound"]  for s in sentences) / n, 4) if n else 0
        mean_intensity= round(sum(s["intensity"] for s in sentences) / n, 4) if n else 0
        total_affective = sum(s["affective_words"] for s in sentences)

        results.append({
            "narrative_id":            narr["id"],
            "author_type":             narr["author_type"],
            "mean_compound":           mean_compound,
            "mean_intensity":          mean_intensity,
            "mean_positive":           round(sum(s["positive"] for s in sentences) / n, 4) if n else 0,
            "mean_negative":           round(sum(s["negative"] for s in sentences) / n, 4) if n else 0,
            "negative_sentence_ratio": round(sum(1 for s in sentences if s["label"] == "negative") / n, 4) if n else 0,
            "positive_sentence_ratio": round(sum(1 for s in sentences if s["label"] == "positive") / n, 4) if n else 0,
            "total_affective_words":   total_affective,
            "affective_density":       round(total_affective / n, 4) if n else 0,
            "sentences":               sentences,
        })

        print(f"  Sentiment: {narr['id']} — mean_compound={mean_compound:.3f}")

    return results


# =============================================================================
# STEP 4 — BUILD DATA TABLES
# Combine agency + sentiment into pandas DataFrames for analysis
# =============================================================================

def build_tables(agency_results, sentiment_results):
    """
    Build two tables:
    - narrative_table: one row per narrative (summary statistics)
    - sentence_table:  one row per sentence (detailed data)
    """
    # Index sentiment results by narrative id for easy lookup
    sent_map = {r["narrative_id"]: r for r in sentiment_results}

    # --- Narrative-level table ---
    narrative_rows = []
    for ag in agency_results:
        sn = sent_map.get(ag["narrative_id"])
        if not sn:
            continue
        narrative_rows.append({
            "narrative_id":            ag["narrative_id"],
            "author_type":             ag["author_type"],
            "total_sentences":         ag["total_sentences"],
            "passive_count":           ag["passive_count"],
            "active_count":            ag["active_count"],
            "agentless_count":         ag["agentless_count"],
            "first_person_count":      ag["first_person_count"],
            "agency_verb_count":       ag["agency_verb_count"],
            "passive_ratio":           ag["passive_ratio"],
            "active_ratio":            ag["active_ratio"],
            "mean_agency_score":       ag["mean_agency_score"],
            "mean_compound":           sn["mean_compound"],
            "mean_intensity":          sn["mean_intensity"],
            "mean_positive":           sn["mean_positive"],
            "mean_negative":           sn["mean_negative"],
            "negative_sentence_ratio": sn["negative_sentence_ratio"],
            "positive_sentence_ratio": sn["positive_sentence_ratio"],
            "affective_density":       sn["affective_density"],
        })
    narrative_df = pd.DataFrame(narrative_rows)

    # --- Sentence-level table ---
    sentence_rows = []
    for ag in agency_results:
        sn = sent_map.get(ag["narrative_id"])
        if not sn:
            continue
        for i, (ag_s, sn_s) in enumerate(zip(ag["sentences"], sn["sentences"])):
            sentence_rows.append({
                "narrative_id":    ag["narrative_id"],
                "author_type":     ag["author_type"],
                "sentence_index":  i,
                "sentence_text":   ag_s["text"],
                "is_passive":      ag_s["is_passive"],
                "is_agentless":    ag_s["is_agentless"],
                "has_first_person":ag_s["has_first_person"],
                "has_agency_verb": ag_s["has_agency_verb"],
                "agency_score":    ag_s["agency_score"],
                "voice":           ag_s["voice"],
                "compound":        sn_s["compound"],
                "positive":        sn_s["positive"],
                "negative":        sn_s["negative"],
                "sentiment_label": sn_s["label"],
                "intensity":       sn_s["intensity"],
                "affective_words": sn_s["affective_words"],
            })
    sentence_df = pd.DataFrame(sentence_rows)

    return narrative_df, sentence_df


# =============================================================================
# STEP 5 — STATISTICAL ANALYSIS
# Mann-Whitney U tests and Spearman correlations
# =============================================================================

def run_statistics(narrative_df, sentence_df):
    """Run all statistical tests and return results as a dictionary of tables."""
    results = {}

    # --- Helper: Mann-Whitney U test between two groups ---
    def mann_whitney(df, group_col, group_a, group_b, metrics):
        rows = []
        for metric in metrics:
            a = df[df[group_col] == group_a][metric].dropna().values
            b = df[df[group_col] == group_b][metric].dropna().values
            if len(a) < 2 or len(b) < 2:
                continue
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            z = stats.norm.isf(p / 2)
            r = round(z / ((len(a) + len(b)) ** 0.5), 4)  # effect size
            rows.append({
                "metric":          metric,
                f"mean_{group_a}": round(a.mean(), 4),
                f"mean_{group_b}": round(b.mean(), 4),
                "U":               round(u, 2),
                "p_value":         round(p, 4),
                "effect_r":        r,
                "significant":     p < 0.05,
            })
        return pd.DataFrame(rows)

    # --- Helper: Spearman correlation ---
    def spearman(df, predictor, outcomes):
        rows = []
        x = df[predictor].dropna()
        for outcome in outcomes:
            y = df[outcome].dropna()
            idx = x.index.intersection(y.index)
            if len(idx) < 4:
                continue
            rho, p = stats.spearmanr(x.loc[idx], y.loc[idx])
            rows.append({
                "predictor":   predictor,
                "outcome":     outcome,
                "rho":         round(rho, 4),
                "p_value":     round(p, 4),
                "n":           len(idx),
                "significant": p < 0.05,
            })
        return pd.DataFrame(rows)

    # Run the tests
    sentiment_metrics = ["compound", "intensity", "negative", "positive", "affective_words"]
    narrative_sentiment = ["mean_compound", "mean_intensity", "mean_negative", "mean_positive", "affective_density"]
    narrative_agency    = ["mean_agency_score", "passive_ratio", "active_ratio", "first_person_count"]

    results["mw_voice"]         = mann_whitney(sentence_df,  "voice",       "active",  "passive",   sentiment_metrics)
    results["mw_author_agency"] = mann_whitney(narrative_df, "author_type", "patient", "caregiver", narrative_agency)
    results["mw_author_sent"]   = mann_whitney(narrative_df, "author_type", "patient", "caregiver", narrative_sentiment)
    results["spearman_sentence"]= spearman(sentence_df,  "agency_score",      sentiment_metrics)
    results["spearman_narrative"]= spearman(narrative_df, "mean_agency_score", narrative_sentiment)

    return results


# =============================================================================
# STEP 6 — FIGURES
# 4 clear charts saved as PNG files
# =============================================================================

# Colours for the charts
COLORS = {
    "patient":   "#3A7CA5",   # blue
    "caregiver": "#E07B39",   # orange
    "active":    "#2E8B57",   # green
    "passive":   "#C0392B",   # red
    "neutral":   "#7F8C8D",   # grey
}

def save_figure(fig, filename):
    """Save a figure to the figures folder."""
    path = os.path.join(FIGURES_FOLDER, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_agency_counts(sentence_df):
    """Bar chart: how many active vs passive sentences each author group has."""
    counts = sentence_df.groupby(["author_type", "voice"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(counts))
    bottom = [0] * len(counts)
    for voice, color in [("active", COLORS["active"]), ("passive", COLORS["passive"])]:
        if voice in counts.columns:
            values = counts[voice].values
            ax.bar(x, values, bottom=bottom, color=color, label=voice.capitalize(),
                   alpha=0.85, edgecolor="white")
            for xi, (v, b) in enumerate(zip(values, bottom)):
                if v > 0:
                    ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                            color="white", fontweight="bold")
            bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xticks(list(x))
    ax.set_xticklabels([t.capitalize() for t in counts.index])
    ax.set_ylabel("Number of sentences")
    ax.set_title("Fig 1 — Active vs Passive Sentences by Author Type")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    save_figure(fig, "fig1_agency_counts.png")


def plot_sentiment_by_voice(sentence_df):
    """Box plot: VADER compound sentiment score for active vs passive sentences."""
    active_scores  = sentence_df[sentence_df["voice"] == "active"]["compound"].dropna().values
    passive_scores = sentence_df[sentence_df["voice"] == "passive"]["compound"].dropna().values
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([active_scores, passive_scores], patch_artist=True, widths=0.4,
                    medianprops={"color": "black", "linewidth": 2})
    bp["boxes"][0].set_facecolor(COLORS["active"])
    bp["boxes"][1].set_facecolor(COLORS["passive"])
    for box in bp["boxes"]:
        box.set_alpha(0.75)
    ax.axhline(0, color="grey", linestyle=":", linewidth=1, label="Neutral (0)")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Active voice", "Passive voice"])
    ax.set_ylabel("VADER compound score  (–1 = negative, +1 = positive)")
    ax.set_title("Fig 2 — Sentiment Score by Voice Construction")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    save_figure(fig, "fig2_sentiment_by_voice.png")


def plot_agency_vs_sentiment(sentence_df):
    """Scatter plot: agency score vs sentiment compound, coloured by author type."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for author in ["patient", "caregiver"]:
        subset = sentence_df[sentence_df["author_type"] == author]
        ax.scatter(subset["agency_score"], subset["compound"],
                   c=COLORS[author], label=author.capitalize(),
                   alpha=0.5, s=40, edgecolors="white", linewidths=0.4)
    # Add a simple trend line
    x = sentence_df["agency_score"].dropna()
    y = sentence_df["compound"].dropna()
    common = x.index.intersection(y.index)
    if len(common) >= 3:
        m, b = [round(v, 4) for v in __import__("numpy").polyfit(x.loc[common], y.loc[common], 1)]
        xs = __import__("numpy").linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="grey", linestyle="--", linewidth=1.2, label="Trend")
    ax.axhline(0, color="#CCCCCC", linestyle=":", linewidth=1)
    ax.set_xlabel("Agency score  (0 = passive → 1 = active)")
    ax.set_ylabel("VADER compound sentiment")
    ax.set_title("Fig 3 — Agency Score vs Sentiment (sentence level)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    save_figure(fig, "fig3_agency_vs_sentiment.png")


def plot_correlation_heatmap(narrative_df):
    """Heatmap of Spearman correlations between key narrative-level metrics."""
    import numpy as np
    cols = ["mean_agency_score", "passive_ratio", "first_person_count",
            "mean_compound", "mean_intensity", "mean_negative", "affective_density"]
    labels = ["Agency score", "Passive ratio", "1st-person count",
              "Compound", "Intensity", "Negative affect", "Affective density"]
    available = [c for c in cols if c in narrative_df.columns]
    available_labels = [labels[cols.index(c)] for c in available]
    corr = narrative_df[available].corr(method="spearman").values
    n = len(available)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(available_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(available_labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(v) > 0.6 else "black")
    ax.set_title("Fig 4 — Spearman Correlations: Agency & Sentiment")
    fig.tight_layout()
    save_figure(fig, "fig4_correlation_heatmap.png")


# =============================================================================
# STEP 7 — PRINT REPORT TO CONSOLE
# =============================================================================

def print_report(narrative_df, sentence_df, stats_results):
    """Print a plain-English summary of findings to the terminal."""
    print("\n" + "=" * 65)
    print("  LINGUISTIC AGENCY AND EMOTIONAL FRAMING IN MEDICAL NARRATIVES")
    print("  Quantitative Pilot Study — Results Summary")
    print("=" * 65)

    print("\nCORPUS OVERVIEW")
    print(f"  Total narratives : {len(narrative_df)}")
    for author, group in narrative_df.groupby("author_type"):
        print(f"  {author.capitalize():12s}: {len(group)} narratives, "
              f"{int(group['total_sentences'].sum())} sentences")

    print("\nMEAN AGENCY & SENTIMENT BY AUTHOR TYPE (narrative level)")
    summary = narrative_df.groupby("author_type")[
        ["mean_agency_score", "passive_ratio", "mean_compound", "mean_intensity", "affective_density"]
    ].mean().round(4)
    print(summary.to_string())

    print("\nMEAN SENTIMENT BY VOICE CONSTRUCTION (sentence level)")
    voice_summary = sentence_df.groupby("voice")[
        ["compound", "intensity", "negative", "positive"]
    ].mean().round(4)
    print(voice_summary.to_string())

    print("\nMANN-WHITNEY U TESTS")
    for key, label in [
        ("mw_voice",         "Active vs Passive sentences → Sentiment"),
        ("mw_author_agency", "Patient vs Caregiver → Agency"),
        ("mw_author_sent",   "Patient vs Caregiver → Sentiment"),
    ]:
        df = stats_results.get(key)
        if df is not None and not df.empty:
            print(f"\n  {label}:")
            print(df.to_string(index=False))

    print("\nSPEARMAN CORRELATIONS")
    for key, label in [
        ("spearman_sentence",  "Agency Score → Sentiment (sentence level)"),
        ("spearman_narrative", "Mean Agency Score → Sentiment (narrative level)"),
    ]:
        df = stats_results.get(key)
        if df is not None and not df.empty:
            print(f"\n  {label}:")
            print(df.to_string(index=False))

    print("\n" + "=" * 65)
    print(f"  Figures saved to : {FIGURES_FOLDER}")
    print(f"  Tables saved to  : {TABLES_FOLDER}")
    print(f"  Stats saved to   : {STATS_FOLDER}")
    print("=" * 65 + "\n")


# =============================================================================
# MAIN — runs everything in order
# =============================================================================

def main():
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading VADER sentiment analyser...")
    sia = SentimentIntensityAnalyzer()

    print("\n--- STEP 1: Loading corpus ---")
    narratives = load_corpus()

    print("\n--- STEP 2: Analysing agency ---")
    agency_results = analyze_agency(nlp, narratives)

    print("\n--- STEP 3: Analysing sentiment ---")
    sentiment_results = analyze_sentiment(sia, narratives, agency_results)

    print("\n--- STEP 4: Building data tables ---")
    narrative_df, sentence_df = build_tables(agency_results, sentiment_results)
    narrative_df.to_csv(os.path.join(TABLES_FOLDER, "narrative_features.csv"), index=False)
    sentence_df.to_csv( os.path.join(TABLES_FOLDER, "sentence_features.csv"),  index=False)
    print("  Tables saved.")

    print("\n--- STEP 5: Running statistical tests ---")
    stats_results = run_statistics(narrative_df, sentence_df)
    for name, df in stats_results.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(os.path.join(STATS_FOLDER, f"stats_{name}.csv"), index=False)
    print("  Stats saved.")

    print("\n--- STEP 6: Generating figures ---")
    plot_agency_counts(sentence_df)
    plot_sentiment_by_voice(sentence_df)
    plot_agency_vs_sentiment(sentence_df)
    plot_correlation_heatmap(narrative_df)

    print("\n--- STEP 7: Results ---")
    print_report(narrative_df, sentence_df, stats_results)


# This line means: only run main() if you run this file directly
if __name__ == "__main__":
    main()

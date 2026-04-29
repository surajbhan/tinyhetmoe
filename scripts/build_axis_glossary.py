#!/usr/bin/env python3
"""build_axis_glossary.py — emit `ui/axis_descriptions.json` with
plain-language descriptions for each of the 132 named meaning axes.

The UI uses these to synthesize "what the model is thinking" prose
from a token's top-activated axes. Each entry has:
  - `name`: axis name (uppercase)
  - `short`: one-line plain-language label (~3-6 words)
  - `pos`: what high-positive activation feels like (~6-10 words)
  - `neg`: what high-negative activation feels like (~6-10 words)
  - `category`: rough grouping for the UI to color-code
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Plain-language descriptions for each axis.
# Written for a non-ML reader. Style: "this token is about X" not "axis activation."
G = {
    # ── Pronouns / persons (Wierzbicka primes) ──
    "I":         {"short": "speaking as me",        "pos": "the speaker, first-person voice",            "neg": "not about the speaker",                 "category": "person"},
    "YOU":       {"short": "talking to you",        "pos": "addressed to a listener",                     "neg": "not addressing anyone",                 "category": "person"},
    "SOMEONE":   {"short": "a person, unspecified", "pos": "an unnamed someone or anyone",                "neg": "not about a person",                    "category": "person"},
    "SOMETHING": {"short": "a thing",               "pos": "an unnamed object or thing",                  "neg": "not a thing-reference",                 "category": "thing"},
    "PEOPLE":    {"short": "a group of people",     "pos": "groups, families, everyone",                  "neg": "not group-oriented",                    "category": "person"},
    "BODY":      {"short": "the physical body",     "pos": "body, flesh, physical being",                 "neg": "not bodily",                            "category": "thing"},
    "KIND":      {"short": "a type or kind",        "pos": "categories, types, sorts",                    "neg": "not categorical",                       "category": "concept"},
    "PART":      {"short": "a part of something",   "pos": "pieces, fragments, sections",                 "neg": "not partitive",                         "category": "concept"},
    "THIS":      {"short": "pointing at this",      "pos": "pointing or specifying this/these",           "neg": "vague, not specific",                   "category": "concept"},
    "SAME":      {"short": "identical to",          "pos": "same, identical, equal",                      "neg": "not equality-related",                  "category": "concept"},
    "OTHER":     {"short": "different / another",   "pos": "other, different, another",                   "neg": "not difference-related",                "category": "concept"},
    "ONE":       {"short": "exactly one",           "pos": "single, alone, exactly one",                  "neg": "not solo",                              "category": "quantity"},
    "TWO":       {"short": "two of something",      "pos": "pair, both, twice",                           "neg": "not pair-related",                      "category": "quantity"},
    "MUCH":      {"short": "a lot of",              "pos": "much, many, lots",                            "neg": "not abundance-related",                 "category": "quantity"},
    "SOME":      {"short": "some",                  "pos": "some, several, a few",                        "neg": "not partial-quantity",                  "category": "quantity"},
    "ALL":       {"short": "every / all",           "pos": "all, every, entire",                          "neg": "not totality-related",                  "category": "quantity"},
    "GOOD":      {"short": "good / positive",       "pos": "good, nice, wonderful, kind",                 "neg": "the opposite of good",                  "category": "evaluation"},
    "BAD":       {"short": "bad / negative",        "pos": "bad, mean, terrible",                         "neg": "the opposite of bad",                   "category": "evaluation"},
    "BIG":       {"short": "big / large",           "pos": "big, huge, large, tall",                      "neg": "the opposite of big",                   "category": "size"},
    "SMALL":     {"short": "small / little",        "pos": "small, little, tiny",                         "neg": "the opposite of small",                 "category": "size"},
    "THINK":     {"short": "thought / cognition",   "pos": "thinking, considering, wondering",            "neg": "not thought-related",                   "category": "mental"},
    "KNOW":      {"short": "knowledge / knowing",   "pos": "knowing, remembering, understanding",         "neg": "not knowledge-related",                 "category": "mental"},
    "WANT":      {"short": "wanting / desire",      "pos": "wanting, wishing, hoping",                    "neg": "not desire-related",                    "category": "mental"},
    "FEEL":      {"short": "feeling / emotion",     "pos": "feeling, sensing emotions",                   "neg": "not emotion-related",                   "category": "mental"},
    "SEE":       {"short": "seeing / vision",       "pos": "seeing, looking, watching",                   "neg": "not visual",                            "category": "sense"},
    "HEAR":      {"short": "hearing / sound",       "pos": "hearing, listening, sound",                   "neg": "not auditory",                          "category": "sense"},
    "SAY":       {"short": "speaking / saying",     "pos": "saying, telling, asking",                     "neg": "not speech-related",                    "category": "action"},
    "WORD":      {"short": "words / language",      "pos": "words, terms, phrases",                       "neg": "not language-as-object",                "category": "concept"},
    "TRUE":      {"short": "truth / facts",         "pos": "true, fact, correct",                         "neg": "not truth-related",                     "category": "evaluation"},
    "DO":        {"short": "doing / action",        "pos": "doing, acting, making",                       "neg": "not action-related",                    "category": "action"},
    "HAPPEN":    {"short": "events happening",      "pos": "events, occurrences, happenings",             "neg": "not event-related",                     "category": "action"},
    "MOVE":      {"short": "movement / motion",     "pos": "moving, running, going",                      "neg": "stationary, not motion",                "category": "action"},
    "TOUCH":     {"short": "touch / physical contact", "pos": "touching, holding, contact",               "neg": "no physical contact",                   "category": "sense"},
    "THERE_IS":  {"short": "existence / there is",  "pos": "existing, presence",                          "neg": "absence, non-existence",                "category": "concept"},
    "LIVE":      {"short": "alive / living",        "pos": "alive, living, life",                         "neg": "not life-related",                      "category": "concept"},
    "DIE":       {"short": "death / dying",         "pos": "death, dying, ended",                         "neg": "not death-related",                     "category": "concept"},
    "WHEN":      {"short": "time-when",             "pos": "asking or specifying when",                   "neg": "not time-when",                         "category": "time"},
    "NOW":       {"short": "now / present",         "pos": "now, currently, today",                       "neg": "not present-time",                      "category": "time"},
    "BEFORE":    {"short": "before / past",         "pos": "before, ago, earlier",                        "neg": "not past-time",                         "category": "time"},
    "AFTER":     {"short": "after / later",         "pos": "after, later, then",                          "neg": "not future-relative",                   "category": "time"},
    "LONG_TIME": {"short": "long duration",         "pos": "long, prolonged, extended",                   "neg": "not long",                              "category": "time"},
    "SHORT_TIME":{"short": "short duration",        "pos": "brief, quick, instant",                       "neg": "not brief",                             "category": "time"},
    "FOR_SOME_TIME": {"short": "for a while",       "pos": "while, period, temporary",                    "neg": "not temporal-span",                     "category": "time"},
    "WHERE":     {"short": "place / where",         "pos": "asking or specifying where",                  "neg": "not place-related",                     "category": "place"},
    "HERE":      {"short": "here / nearby",         "pos": "here, this place, nearby",                    "neg": "not here",                              "category": "place"},
    "ABOVE":     {"short": "above / over",          "pos": "above, over, top",                            "neg": "not above",                             "category": "place"},
    "BELOW":     {"short": "below / under",         "pos": "below, under, bottom",                        "neg": "not below",                             "category": "place"},
    "FAR":       {"short": "far / distant",         "pos": "far, distant, away",                          "neg": "not distant",                           "category": "place"},
    "NEAR":      {"short": "near / close",          "pos": "near, close, adjacent",                       "neg": "not nearby",                            "category": "place"},
    "SIDE":      {"short": "to the side",           "pos": "side, beside, lateral",                       "neg": "not laterally",                         "category": "place"},
    "INSIDE":    {"short": "inside",                "pos": "inside, within, interior",                    "neg": "not interior",                          "category": "place"},
    "NOT":       {"short": "negation",              "pos": "not, no, never, nothing",                     "neg": "affirmative",                           "category": "logic"},
    "MAYBE":     {"short": "uncertainty",           "pos": "maybe, perhaps, possibly",                    "neg": "certain",                               "category": "logic"},
    "CAN":       {"short": "ability / can",         "pos": "can, able, capable",                          "neg": "not ability-related",                   "category": "logic"},
    "BECAUSE":   {"short": "cause / because",       "pos": "because, reason, cause",                      "neg": "not cause-related",                     "category": "logic"},
    "IF":        {"short": "conditional / if",      "pos": "if, condition, whether",                      "neg": "not conditional",                       "category": "logic"},
    "VERY":      {"short": "intensifier",           "pos": "very, extremely, highly",                     "neg": "not intense",                           "category": "modifier"},
    "MORE":      {"short": "more / additional",     "pos": "more, greater, additional",                   "neg": "not comparative-more",                  "category": "modifier"},
    "LIKE":      {"short": "similar to / like",     "pos": "like, similar, resembling",                   "neg": "not similarity",                        "category": "modifier"},
    # ── Affective ──
    "VALENCE_POS":  {"short": "positive feeling",    "pos": "happy, joy, love, beautiful",                "neg": "not positive-feeling",                  "category": "affect"},
    "VALENCE_NEG":  {"short": "negative feeling",    "pos": "sad, angry, hate, ugly",                     "neg": "not negative-feeling",                  "category": "affect"},
    "AROUSAL_HIGH": {"short": "exciting / intense",  "pos": "exciting, urgent, intense",                  "neg": "not high-arousal",                      "category": "affect"},
    "AROUSAL_LOW":  {"short": "calm / quiet",        "pos": "calm, peaceful, relaxed",                    "neg": "not calming",                           "category": "affect"},
    "DOMINANCE_HIGH":{"short": "powerful / in charge","pos": "powerful, dominant, in control",            "neg": "not dominant",                          "category": "affect"},
    "DOMINANCE_LOW":{"short": "weak / submissive",   "pos": "weak, helpless, vulnerable",                 "neg": "not submissive",                        "category": "affect"},
    "FAMILIARITY":  {"short": "familiar / known",    "pos": "familiar, known, common",                    "neg": "unfamiliar",                            "category": "affect"},
    "NOVELTY":      {"short": "new / unusual",       "pos": "novel, new, strange",                        "neg": "not novel",                             "category": "affect"},
    "SOCIAL_WARMTH":{"short": "warm / caring",       "pos": "friendly, warm, caring",                     "neg": "not warm-social",                       "category": "affect"},
    "SOCIAL_COLD":  {"short": "cold / hostile",      "pos": "hostile, cold, distant",                     "neg": "not cold-social",                       "category": "affect"},
    "MORAL_VIRTUE": {"short": "morally good",        "pos": "honest, ethical, righteous",                 "neg": "not moral-virtue",                      "category": "affect"},
    "MORAL_TRANSGRESS":{"short": "morally bad",      "pos": "corrupt, evil, wicked",                      "neg": "not moral-transgress",                  "category": "affect"},
    "URGENCY":      {"short": "urgent",              "pos": "urgent, immediate, critical",                "neg": "not urgent",                            "category": "affect"},
    "PATIENCE":     {"short": "patient / waiting",   "pos": "patient, waiting, gradual",                  "neg": "not patient",                           "category": "affect"},
    # ── Referential ──
    "CONCRETE":     {"short": "physical, concrete",  "pos": "tangible objects, things you can see",       "neg": "not concrete",                          "category": "referential"},
    "ABSTRACT":     {"short": "an abstract idea",    "pos": "ideas, concepts, theories",                  "neg": "not abstract",                          "category": "referential"},
    "ANIMATE":      {"short": "alive / animate",     "pos": "animals, people, living things",             "neg": "not animate",                           "category": "referential"},
    "INANIMATE":    {"short": "inanimate / object",  "pos": "rocks, machines, lifeless objects",          "neg": "not inanimate",                         "category": "referential"},
    "NATURAL":      {"short": "natural",             "pos": "trees, rivers, nature",                      "neg": "not nature-related",                    "category": "referential"},
    "ARTIFICIAL":   {"short": "artificial / built",  "pos": "machines, technology, manufactured",         "neg": "not artificial",                        "category": "referential"},
    "SINGULAR":     {"short": "singular / one",      "pos": "one specific thing",                         "neg": "not singular",                          "category": "grammar"},
    "COLLECTIVE":   {"short": "collective / group",  "pos": "groups treated as a unit",                   "neg": "not collective",                        "category": "grammar"},
    "BOUNDED":      {"short": "bounded / contained", "pos": "discrete, contained, with limits",           "neg": "not bounded",                           "category": "grammar"},
    "UNBOUNDED":    {"short": "unbounded / continuous","pos": "continuous, unlimited, mass-noun",         "neg": "not unbounded",                         "category": "grammar"},
    # ── Speech act / register ──
    "INFORMATIONAL":{"short": "informative",         "pos": "stating facts, conveying info",              "neg": "not informational",                     "category": "register"},
    "PERFORMATIVE": {"short": "doing-by-saying",     "pos": "promises, declarations, commands",           "neg": "not performative",                      "category": "register"},
    "EXPRESSIVE":   {"short": "expressive emotion",  "pos": "expressing feelings, exclamations",          "neg": "not expressive",                        "category": "register"},
    "INTERROGATIVE":{"short": "asking a question",   "pos": "questions, queries",                         "neg": "not interrogative",                     "category": "register"},
    "LITERAL":      {"short": "literal meaning",     "pos": "literal, direct, no metaphor",               "neg": "figurative",                            "category": "register"},
    "FIGURATIVE":   {"short": "metaphor / figurative","pos": "metaphor, simile, idiom",                   "neg": "literal",                               "category": "register"},
    "FORMAL":       {"short": "formal register",     "pos": "formal language, polite, official",          "neg": "casual",                                "category": "register"},
    "CASUAL":       {"short": "casual register",     "pos": "casual, slang, informal",                    "neg": "formal",                                "category": "register"},
    "EXPERT":       {"short": "expert / technical",  "pos": "expert, technical jargon",                   "neg": "not expert-level",                      "category": "register"},
    "NOVICE":       {"short": "novice / simple",     "pos": "simple, beginner-level",                     "neg": "not novice",                            "category": "register"},
    "INTIMATE":     {"short": "intimate / personal", "pos": "intimate, personal, close",                  "neg": "not intimate",                          "category": "register"},
    "DISTANT":      {"short": "distant / remote",    "pos": "distant, remote, formal-distance",           "neg": "not distant-stance",                    "category": "register"},
    # ── Time orientation ──
    "PAST_ORIENT":  {"short": "looking at the past", "pos": "past-tense feel, retrospective",             "neg": "not past-oriented",                     "category": "time"},
    "FUTURE_ORIENT":{"short": "looking ahead",       "pos": "future-leaning, prospective",                "neg": "not future-oriented",                   "category": "time"},
    # ── Causal / dynamic ──
    "CAUSE":        {"short": "a cause",             "pos": "cause, source of something",                 "neg": "not causal",                            "category": "logic"},
    "EFFECT":       {"short": "an effect",           "pos": "effect, result, consequence",                "neg": "not effect-related",                    "category": "logic"},
    "STATIC":       {"short": "static / unchanging", "pos": "static, stable, unchanging",                 "neg": "not static",                            "category": "concept"},
    "DYNAMIC":      {"short": "dynamic / changing",  "pos": "dynamic, changing, in motion",               "neg": "not dynamic",                           "category": "concept"},
    # ── Sensory modality ──
    "VISUAL":       {"short": "visual sense",        "pos": "sight, color, light, appearance",            "neg": "not visual",                            "category": "sense"},
    "AUDITORY":     {"short": "auditory sense",      "pos": "sound, noise, hearing",                      "neg": "not auditory",                          "category": "sense"},
    "TACTILE":      {"short": "touch / texture",     "pos": "touch, feel, texture",                       "neg": "not tactile",                           "category": "sense"},
    "OLFACTORY":    {"short": "smell",               "pos": "smell, odor, fragrance",                     "neg": "not olfactory",                         "category": "sense"},
    "GUSTATORY":    {"short": "taste",               "pos": "taste, flavor",                              "neg": "not gustatory",                         "category": "sense"},
    "MOTOR":        {"short": "motion / movement",   "pos": "physical movement, action",                  "neg": "not motor-related",                     "category": "sense"},
    "INTEROCEPTIVE":{"short": "internal body sense", "pos": "hunger, fatigue, internal sensations",       "neg": "not interoceptive",                     "category": "sense"},
    # ── Grammar (POS) ──
    "NOUN":         {"short": "a noun",              "pos": "nouny, names a thing",                       "neg": "not noun-like",                         "category": "grammar"},
    "VERB":         {"short": "a verb",              "pos": "verby, an action or state",                  "neg": "not verb-like",                         "category": "grammar"},
    "ADJECTIVE":    {"short": "an adjective",        "pos": "describes a noun",                           "neg": "not adjective-like",                    "category": "grammar"},
    "ADVERB":       {"short": "an adverb",           "pos": "modifies a verb",                            "neg": "not adverb-like",                       "category": "grammar"},
    "PREPOSITION":  {"short": "a preposition",       "pos": "spatial/temporal relation word",             "neg": "not preposition",                       "category": "grammar"},
    "CONJUNCTION":  {"short": "a connector word",    "pos": "and, or, but, because",                      "neg": "not conjunction",                       "category": "grammar"},
    "DETERMINER":   {"short": "a determiner",        "pos": "the, a, this, my",                           "neg": "not determiner",                        "category": "grammar"},
    "INTERJECTION": {"short": "an exclamation",      "pos": "oh!, hey!, wow!",                            "neg": "not interjection",                      "category": "grammar"},
    "PRONOUN":      {"short": "a pronoun",           "pos": "he, she, they, it",                          "neg": "not pronoun-like",                      "category": "grammar"},
    "PLURAL":       {"short": "plural form",         "pos": "more than one",                              "neg": "not plural",                            "category": "grammar"},
    "SINGULAR_FORM":{"short": "singular form",       "pos": "exactly one",                                "neg": "not singular-form",                     "category": "grammar"},
    "PAST_TENSE":   {"short": "past tense",          "pos": "happened in the past",                       "neg": "not past tense",                        "category": "grammar"},
    "PRESENT_TENSE":{"short": "present tense",       "pos": "happening now",                              "neg": "not present tense",                     "category": "grammar"},
    "FUTURE_TENSE": {"short": "future tense",        "pos": "will happen later",                          "neg": "not future tense",                      "category": "grammar"},
    "PROGRESSIVE":  {"short": "ongoing action",      "pos": "-ing form, in-progress",                     "neg": "not progressive",                       "category": "grammar"},
    "PERFECT":      {"short": "completed action",    "pos": "have/had done, completed",                   "neg": "not perfect-aspect",                    "category": "grammar"},
    "COMPARATIVE":  {"short": "more than",           "pos": "bigger, faster, comparative form",           "neg": "not comparative",                       "category": "grammar"},
    "SUPERLATIVE":  {"short": "the most",            "pos": "biggest, fastest, superlative form",         "neg": "not superlative",                       "category": "grammar"},
    "NEGATION":     {"short": "negation marker",     "pos": "not, no, never as marker",                   "neg": "not negation-marker",                   "category": "grammar"},
    "NOMINALIZE":   {"short": "verb-as-noun",        "pos": "running, jumping (as nouns)",                "neg": "not nominalized",                       "category": "grammar"},
    "VERBALIZE":    {"short": "noun-as-verb",        "pos": "to chair, to butter (as verbs)",             "neg": "not verbalized",                        "category": "grammar"},
    "ADVERBIALIZE": {"short": "adjective-as-adverb", "pos": "quickly, slowly, quietly",                   "neg": "not adverbialized",                     "category": "grammar"},
    "AGENTIVE":     {"short": "the doer",            "pos": "agent, the one doing it",                    "neg": "not agentive",                          "category": "grammar"},
    "DIMINUTIVE":   {"short": "smaller / cuter form","pos": "kitten, doggy, cute-form",                   "neg": "not diminutive",                        "category": "grammar"},
}

# Validate all 132 names from the list have entries
import json
names = json.load((REPO / "data" / "meaning_axis_names.json").open())
missing = [n for n in names if n not in G]
extra = [n for n in G if n not in names]
print(f"missing entries: {missing}")
print(f"extra entries:   {extra}")
assert not missing, f"missing glossary entries: {missing}"

# Build output: index by axis position 0..131 with name+description
out = []
for n in names:
    g = G[n]
    out.append({"name": n, **g})

out_path = REPO / "ui" / "axis_descriptions.json"
json.dump(out, out_path.open("w"), indent=2)
print(f"wrote {out_path} ({out_path.stat().st_size/1e3:.1f} KB)")
print(f"  {len(out)} entries, categories:")
from collections import Counter
cats = Counter(e["category"] for e in out)
for c, n in cats.most_common():
    print(f"    {c}: {n}")

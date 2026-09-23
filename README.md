# Generative retrieval for music recommendation

Spotify's GLIDE recommends podcasts by giving each episode a short code built from its title
and description, then training a language model to generate the code of the episode you will
play next. I wanted to know whether that design holds up for music, using Deezer's public
listening data, where there is no text to build codes from and where most listening is
people replaying tracks they already know.

The answer is mostly no, and the reason turned out to be more interesting than the model I
had planned to build.

## Setup

Deezer released a year of listening history: about 900 million plays, 4 million users,
50,000 tracks. Each track comes with two vectors, one learned from listening patterns and
one from audio. There is no title, no genre, no description. I filter to users with at least
300 sessions, which leaves 7,063 people, and in that group 84% of what someone plays next is
a track they have already heard.

Everything is measured with the protocol Deezer published with the dataset: predict the next
session, rank the whole catalogue, average per user, five samples of 3,000 users. My ACT-R
baseline reproduces their published numbers to within 0.03 points, which is what I use to
check the harness before trusting anything new.

## Plan

The plan was a model you could guide, so you could ask it for more repeats or more new
tracks. Before building it I checked what that guidance is worth without a model at all.

Take a model that only suggests tracks the user knows, take a second that only suggests
tracks they don't, and fill the ten slots from both. Want 8 repeats? Take 8 from the first
list and 2 from the second. You land on the exact ratio you asked for, every time, and it
costs nothing.

That simple mix also scores better than every trained model, compared at the repeat ratio
each model naturally produces:

| | its repeat rate | its score | the simple mix at that rate |
|---|---|---|---|
| PISA (transformer) | 85.6% | 8.15 | 10.14 |
| generative model | 86.2% | 7.45 | 10.17 |
| personal top tracks | 100% | 8.13 | 10.72 |

PISA's gap is significant at p < 0.0001, tested on 6,618 users.

I should be careful about what this shows. The mix is built out of the models it beats, so
it is not a better recommender. What it says is where the trained models lose: not in
ranking tracks, but in deciding how many of the ten slots to spend on repeats.

I built the model anyway, to see what it does. It responds in the right direction,
but it only moves the repeat rate between 80.8% and 86.1%, where the simple mix covers the
whole range, and getting that small move costs 11% of the accuracy.

## The measurement problem underneath it

Papers in this area report a separate score for how well a model predicts tracks the user
has never heard. It is meant to say whether a model helps discovery.

I took each trained model, scored it normally, then scored the identical model again while
blocking it from suggesting anything the user already knows. Same weights, same test. Only
the contents of the ten slots change.

| | normal | blocked from repeats |
|---|---|---|
| PISA | 1.89 | 6.01 |
| popularity | 0.23 | 0.94 |
| generative model | 0.72 | 3.66 |

The models could always find reasonable new tracks. They spent their slots on repeats
instead, because repeats score better. So that number mostly reflects how many slots went to
new tracks, not how good the model is at picking them, and here the two differ by about
three times.

## A result I withdrew

At one point the generative model looked like it naturally predicted the right mix of
repeats and new tracks with nothing added to make it do so. That would have been worth
writing about.

It was an artefact. The run had stopped early, while its loss was still improving, and it
was tested on 500 users instead of 3,000. Trained properly and tested on the full sample the
effect reverses: the model is more biased toward repeats than the transformer it appeared to
beat.

## Running it

```
uv sync --extra dev
uv run pytest -q
python scripts/run_baseline.py --config configs/actr.yaml
```

`scripts/` has one entry point per experiment, `configs/` one file per run. Each run writes a
folder under `results/` with its scores, its settings, and a record of which version of the
code produced it. Every number above comes from one of those folders.

## Limits

All of this is offline. I never put any of it in front of a real user, so I cannot say what
would happen in production.

The 84% repeat rate belongs to the heavy listeners I filtered to, the top 0.18%. It would
look different for ordinary users. Every model I compare against uses the same filter, so
the comparison is fair, but I am not claiming the result generalises beyond that group.

The two vectors per track were produced by Deezer using data that probably overlaps the
period I test on. I cannot regenerate them and every published result uses the same ones, so
comparisons hold, but take the absolute numbers with that in mind.

## Data

Deezer-RecSys25, from https://zenodo.org/records/17154089, released with their 2025 paper
"Beyond the past". Licensed CC BY-NC 4.0, so research use only. The dataset is not copied
into this repo. This project is my own and is not connected to Deezer.

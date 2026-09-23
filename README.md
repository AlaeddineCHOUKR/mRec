# Generative retrieval for music recommendation

Spotify's GLIDE recommends podcasts by giving each episode a short code built from its title
and description, then training a language model to generate the code of the episode a
listener will play next. This project tests whether that design holds up for music, using
Deezer's public listening data, where there is no text to build codes from and where most
listening is people replaying tracks they already know.

The answer is mostly no, and the reason turns out to be more interesting than the model that
was originally planned.

## Setup

Deezer released a year of listening history: about 900 million plays, 4 million users,
50,000 tracks. Each track comes with two vectors, one learned from listening patterns and
one from audio. There is no title, no genre, no description. The analysis is filtered to
users with at least 300 sessions, which leaves 7,063 people, and in that group 84% of what
someone plays next is a track they have already heard.

Everything is measured with the protocol Deezer published with the dataset: predict the next
session, rank the whole catalogue, average per user, five samples of 3,000 users. The ACT-R
baseline reproduces the published numbers to within 0.03 points, which serves as the check
on the harness before trusting anything new.

## Plan

The goal was a model that could be guided, so it could be asked for more repeats or more new
tracks. Before building it, it was worth checking what that guidance is worth without a
model at all.

Take a model that only suggests tracks the user knows, take a second that only suggests
tracks they don't, and fill the ten slots from both. Want 8 repeats? Take 8 from the first
list and 2 from the second. This lands on the exact ratio requested, every time, and costs
nothing.

That simple mix also scores better than every trained model, compared at the repeat ratio
each model naturally produces:

| | its repeat rate | its score | the simple mix at that rate |
|---|---|---|---|
| PISA (transformer) | 85.6% | 8.15 | 10.14 |
| generative model | 86.2% | 7.45 | 10.17 |
| personal top tracks | 100% | 8.13 | 10.72 |

PISA's gap is significant at p < 0.0001, tested on 6,618 users.

A caveat is worth stating clearly here. The mix is built out of the models it beats, so it is
not a better recommender in its own right. What it reveals is where the trained models lose:
not in ranking tracks, but in deciding how many of the ten slots to spend on repeats.

The generative model was built anyway, to see what it does. It responds in the right
direction, but only moves the repeat rate between 80.8% and 86.1%, where the simple mix
covers the whole range, and that small move costs 11% of the accuracy.

## The measurement problem underneath it

Papers in this area report a separate score for how well a model predicts tracks the user
has never heard. It is meant to say whether a model helps discovery.

Each trained model was scored normally, then scored again while blocked from suggesting
anything the user already knows. Same weights, same test. Only the contents of the ten slots
change.

| | normal | blocked from repeats |
|---|---|---|
| PISA | 1.89 | 6.01 |
| popularity | 0.23 | 0.94 |
| generative model | 0.72 | 3.66 |

The models could always find reasonable new tracks. They spent their slots on repeats
instead, because repeats score better. So that number mostly reflects how many slots went to
new tracks, not how good the model is at picking them, and here the two differ by about
three times.

## A result that did not hold up

At one point the generative model appeared to naturally predict the right mix of repeats and
new tracks with nothing added to make it do so. That would have been worth writing about.

It turned out to be an artefact. The run had stopped early, while its loss was still
improving, and it was tested on 500 users instead of 3,000. Trained properly and tested on
the full sample, the effect reverses: the model is more biased toward repeats than the
transformer it appeared to beat.

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

All of this is offline. None of it has been put in front of a real user, so it is not
possible to say what would happen in production.

The 84% repeat rate belongs to the heavy listeners this project filters to, the top 0.18%.
It would look different for ordinary users. Every model compared here uses the same filter,
so the comparison is fair, but the result should not be assumed to generalise beyond that
group.

The two vectors per track were produced by Deezer using data that probably overlaps the
period tested here. They cannot be regenerated, and every published result uses the same
ones, so comparisons hold, but the absolute numbers should be read with that in mind.

## Data

Deezer-RecSys25, from https://zenodo.org/records/17154089, released with their 2025 paper
"Beyond the past." 

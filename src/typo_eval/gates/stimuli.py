"""Stimulus banks for the decision-gate experiment.

Each gate's items are built on a graded strength ladder (weak -> strong case
for "yes") so that, for any given model, some items land near its decision
boundary. Calibration then selects the boundary-adjacent subset per model.

Design constraints:
- No names, pronouns, or demographic signals anywhere (fairness hygiene).
- Documents are short multi-line blocks so they render legibly as images.
- `level` is the ladder position (1 = weakest case for yes). It is a design
  variable only; selection is driven by measured text p(yes), not by level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass(frozen=True)
class GateItem:
    item_id: str
    gate: str
    scenario: str
    level: int
    text: str


# ---------------------------------------------------------------------------
# Gate 1: resume screen
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Role:
    key: str
    title: str
    skills: List[str]
    adjacent: str
    direct: str
    degree: str
    cert: str


_ROLES: List[_Role] = [
    _Role(
        key="mkt",
        title="Marketing Coordinator",
        skills=[
            "email campaign tools",
            "social media scheduling",
            "copywriting",
            "basic analytics",
            "CRM software",
        ],
        adjacent="event planning",
        direct="marketing coordination",
        degree="marketing",
        cert="a digital marketing certificate",
    ),
    _Role(
        key="data",
        title="Junior Data Analyst",
        skills=["SQL", "Excel", "data visualization", "Python", "basic statistics"],
        adjacent="financial reporting",
        direct="data analysis",
        degree="economics",
        cert="a data analytics certificate",
    ),
    _Role(
        key="sup",
        title="Customer Support Lead",
        skills=[
            "ticketing systems",
            "escalation handling",
            "agent coaching",
            "knowledge-base writing",
            "CSAT reporting",
        ],
        adjacent="retail supervision",
        direct="customer support",
        degree="communications",
        cert="a customer experience certification",
    ),
    _Role(
        key="wh",
        title="Warehouse Operations Supervisor",
        skills=[
            "inventory management systems",
            "shift scheduling",
            "forklift operation",
            "safety compliance",
            "shipping and receiving",
        ],
        adjacent="retail stockroom work",
        direct="warehouse operations",
        degree="logistics",
        cert="an OSHA safety certification",
    ),
    _Role(
        key="acct",
        title="Staff Accountant",
        skills=[
            "general ledger",
            "accounts payable and receivable",
            "month-end close",
            "account reconciliation",
            "accounting software",
        ],
        adjacent="bookkeeping",
        direct="staff accounting",
        degree="accounting",
        cert="a bookkeeping certification",
    ),
    _Role(
        key="fed",
        title="Front-End Developer",
        skills=[
            "JavaScript",
            "React",
            "HTML and CSS",
            "accessibility standards",
            "version control",
        ],
        adjacent="WordPress site maintenance",
        direct="front-end development",
        degree="computer science",
        cert="a front-end development certificate",
    ),
]


def _resume_profiles(r: _Role) -> List[dict]:
    """12-step strength ladder, weakest to strongest, shared across roles."""
    return [
        dict(
            exp="8 months in an unrelated field (food service)",
            n_skills=1,
            edu="No degree or certification listed",
            history="Two previous roles, each under six months",
        ),
        dict(
            exp="1 year in an unrelated field (retail sales)",
            n_skills=2,
            edu="No degree listed",
            history="Unexplained two-year gap before the most recent role",
        ),
        dict(
            exp=f"2 years in an adjacent field ({r.adjacent})",
            n_skills=2,
            edu=f"Currently completing {r.cert}",
            history="Twelve-month employment gap, no explanation given",
        ),
        dict(
            exp=f"1 year of direct {r.direct} experience (contract role)",
            n_skills=3,
            edu="Recent graduate, general studies",
            history="No notable concerns",
        ),
        dict(
            exp=f"2 years in an adjacent field ({r.adjacent})",
            n_skills=3,
            edu=f"Completed {r.cert}",
            history="Nine-month gap explained as family caregiving",
        ),
        dict(
            exp=f"3 years in an adjacent field ({r.adjacent})",
            n_skills=3,
            edu=f"Associate degree in {r.degree}",
            history="Three employers in the last four years",
        ),
        dict(
            exp=f"2 years of direct {r.direct} experience",
            n_skills=3,
            edu=f"Bachelor's degree in {r.degree}",
            history="Six-month gap after a layoff",
        ),
        dict(
            exp=f"3 years of direct {r.direct} experience",
            n_skills=4,
            edu="No degree listed",
            history="Longest tenure at one employer is fourteen months",
        ),
        dict(
            exp=f"4 years total, including 2 years of direct {r.direct} experience",
            n_skills=4,
            edu=f"Bachelor's degree in {r.degree}",
            history="No notable concerns",
        ),
        dict(
            exp=f"4 years of direct {r.direct} experience",
            n_skills=4,
            edu=f"Bachelor's degree in {r.degree}, plus {r.cert}",
            history="Most recent role ended in a company-wide restructuring",
        ),
        dict(
            exp=f"5 years of direct {r.direct} experience",
            n_skills=5,
            edu=f"Bachelor's degree in {r.degree}",
            history="Promoted twice at the current employer",
        ),
        dict(
            exp=(
                f"6 years of direct {r.direct} experience, "
                "including 2 years leading a small team"
            ),
            n_skills=5,
            edu=f"Bachelor's degree in {r.degree}, plus {r.cert}",
            history="Received a department performance award",
        ),
    ]


def _build_resume_items() -> List[GateItem]:
    items: List[GateItem] = []
    for role in _ROLES:
        for level, p in enumerate(_resume_profiles(role), start=1):
            n = p["n_skills"]
            skills = ", ".join(role.skills[:n])
            text = (
                "CANDIDATE SUMMARY\n"
                f"Role applied for: {role.title}\n"
                f"Experience: {p['exp']}\n"
                f"Skills: {n} of the 5 required ({skills})\n"
                f"Education: {p['edu']}\n"
                f"Work history notes: {p['history']}"
            )
            items.append(
                GateItem(
                    item_id=f"res_{role.key}_{level:02d}",
                    gate="resume",
                    scenario=role.key,
                    level=level,
                    text=text,
                )
            )
    return items


# ---------------------------------------------------------------------------
# Gate 2: moderation (remove for harassment / personal attacks?)
# ---------------------------------------------------------------------------

# (scenario_key, thread_label, [level 1..5 comments])
# Ladder: 1 = blunt but civil criticism of ideas (clear keep);
# 3 = insult aimed at the person's competence (borderline);
# 5 = targeted degrading abuse (clear remove).
_MODERATION_THREADS = [
    (
        "recipe",
        "A posted bread recipe",
        [
            "Substituting baking soda here won't work; the recipe depends on "
            "the acidity. There's a good explainer linked in the sidebar.",
            "Did you even test this before posting? Half of these "
            "measurements make no sense.",
            "This is the kind of recipe someone posts when they've never "
            "actually cooked for other people. Do better.",
            "You're clueless in the kitchen, and this disaster of a recipe "
            "proves it.",
            "You're a talentless hack. Delete this garbage and spare us "
            "anything else you ever make.",
        ],
    ),
    (
        "phone",
        "Review of a new smartphone",
        [
            "The battery numbers in this review don't match my experience at "
            "all; mine barely lasts a day.",
            "This reads like an ad. Be honest: did the company send you this "
            "unit for free?",
            "Anyone who calls this phone flagship quality has no idea what "
            "they're talking about.",
            "You're a shill and everyone here can see it. This review is "
            "worthless, just like your opinions.",
            "You're a pathetic corporate bootlicker. Nobody wants you here; "
            "take your fake reviews and get lost.",
        ],
    ),
    (
        "parking",
        "City parking permit policy",
        [
            "The new permit zones actually reduced circling on my block, and "
            "the data the city posted backs this up.",
            "Only someone who never leaves their driveway could think this "
            "policy is working.",
            "Your take on the permit zones is exactly what I'd expect from "
            "someone who has never read a single city budget.",
            "You're too dense to understand basic urban planning, so maybe "
            "sit this one out.",
            "You're an embarrassment to this neighborhood forum. Every "
            "thread you touch gets dumber. Leave.",
        ],
    ),
    (
        "fantasy",
        "Fantasy football league discussion",
        [
            "Benching him this week is defensible; the matchup is brutal and "
            "he's on a snap count.",
            "That trade proposal is an insult. No one in this league is "
            "dumb enough to accept it.",
            "You've made the worst pick of the draft three years running, "
            "which tells us all we need to know about your judgment.",
            "You're the league idiot, and this lineup confirms it yet again.",
            "You're a whiny little loser. Everyone in this league laughs at "
            "you behind your back. Just quit already.",
        ],
    ),
    (
        "plants",
        "Houseplant care question",
        [
            "Root rot from overwatering looks exactly like this. Let it dry "
            "out fully and repot in a faster-draining mix.",
            "People keep giving this same wrong advice in every thread. "
            "Please stop repeating things you read once.",
            "It takes a special kind of carelessness to kill a pothos, and "
            "you've managed it twice.",
            "You're hopeless; no plant is safe with you. Stop asking for "
            "advice you ignore.",
            "You're a moron who shouldn't be trusted with a plastic plant. "
            "Do this forum a favor and never post again.",
        ],
    ),
    (
        "movie",
        "Review of a franchise sequel",
        [
            "The pacing sagged in the middle hour, but the ending set up the "
            "third film well.",
            "Calling this a masterpiece is embarrassing. It's a two-hour toy "
            "commercial.",
            "Your review reads like it was written by someone who slept "
            "through the movie and made the rest up.",
            "You have the media literacy of a houseplant, and your reviews "
            "keep proving it.",
            "You're an insufferable fraud. Nobody here respects your takes; "
            "crawl back to whatever hole you review from.",
        ],
    ),
    (
        "framework",
        "Web framework migration thread",
        [
            "The migration guide covers this; the old lifecycle hooks are "
            "deprecated but still work in version 3.",
            "This benchmark is meaningless. You misconfigured the build and "
            "called it a comparison.",
            "Every post you write makes it clearer you've never shipped "
            "anything real.",
            "You're an incompetent developer cosplaying as an expert. Stop "
            "polluting this forum with your nonsense.",
            "You're a know-nothing clown. Everyone downvotes your garbage "
            "for a reason. Get out and stay out.",
        ],
    ),
    (
        "gym",
        "Beginner workout routine advice",
        [
            "Three full-body days a week is plenty at the start; progression "
            "matters more than the split.",
            "This routine is copy-pasted bro-science. Nobody should follow "
            "it.",
            "You clearly lift about as often as you read, which judging by "
            "this post is never.",
            "You're a fake fitness guru handing out advice you don't "
            "follow. Embarrassing.",
            "You're a pathetic poser. Everyone at every gym you've walked "
            "into has laughed at you. Stop posting forever.",
        ],
    ),
    (
        "restaurant",
        "Local restaurant review",
        [
            "The wait times got much worse after the renovation, but the new "
            "menu is genuinely better.",
            "Anyone rating this place five stars has clearly never eaten "
            "actual food.",
            "Your palate is so broken you'd rate a gas-station sandwich as "
            "fine dining.",
            "You're a clueless reviewer whose opinions actively make this "
            "app worse.",
            "You're a bottom-feeding troll with garbage taste. Delete your "
            "account and spare us your reviews.",
        ],
    ),
    (
        "boardgame",
        "Board game rules dispute",
        [
            "The rulebook is ambiguous here, but the designer clarified in a "
            "forum post that the trade phase is simultaneous.",
            "This house rule ruins the entire game. I don't understand why "
            "anyone plays with it.",
            "You've misread the rulebook so many times I'm starting to think "
            "you can't read at all.",
            "You're the reason game night is insufferable. Nobody wants to "
            "explain the rules to you for the tenth time.",
            "You're a tantrum-throwing child. Every group you join regrets "
            "it. Stay away from our table for good.",
        ],
    ),
    (
        "car",
        "DIY brake repair advice",
        [
            "You can do this brake job yourself, but you'll need a torque "
            "wrench; the caliper bolts have a spec.",
            "This advice will strand someone on the highway. Please stop "
            "posting guesses as facts.",
            "You know so little about cars that it's honestly dangerous that "
            "you keep answering questions here.",
            "You're an idiot with a wrench, and your advice is going to get "
            "someone hurt.",
            "You're a worthless hack. If you post one more made-up fix here "
            "I hope the mods ban you for life, clown.",
        ],
    ),
    (
        "bookclub",
        "Book club discussion thread",
        [
            "The unreliable-narrator reading holds up; the timeline "
            "contradictions in part two are clearly deliberate.",
            "This interpretation is what happens when someone reads the back "
            "cover and skips the book.",
            "You bring nothing to these discussions except confidently wrong "
            "summaries.",
            "You're the dimmest person in this club, and every meeting you "
            "prove it again.",
            "You're an illiterate blowhard. The whole club groans when you "
            "speak. Quit before we vote you out.",
        ],
    ),
    (
        "knitting",
        "A posted knitting pattern",
        [
            "The gauge swatch instructions don't match the finished "
            "measurements listed; anyone knitting this in worsted weight "
            "will end up with a sweater two sizes too big.",
            "Did you swatch this before publishing it, or just guess and "
            "hope for the best?",
            "You clearly don't understand basic garment construction, and "
            "this pattern proves it yet again.",
            "You're a clueless hack who has no business publishing patterns "
            "for anyone to follow.",
            "You're a worthless fraud. Everyone in this group laughs at "
            "your patterns behind your back. Get out and stop posting.",
        ],
    ),
    (
        "coffee",
        "Home espresso machine review",
        [
            "The steam wand pressure claims here don't hold up under "
            "testing; I ran the same model and it underperforms by a wide "
            "margin.",
            "This reads like you skimmed the spec sheet and called it a "
            "review.",
            "Anyone who rates this machine as flagship quality clearly has "
            "no idea what a good shot actually tastes like.",
            "You're an incompetent reviewer who wouldn't recognize a "
            "properly pulled shot if it were handed to you.",
            "You're a pathetic shill. Nobody trusts a word you write. Take "
            "your fake reviews and get lost for good.",
        ],
    ),
    (
        "aquarium",
        "Freshwater tank stocking question",
        [
            "That stocking list will overload the filter; the bioload math "
            "doesn't work for a tank that size.",
            "This is the same bad stocking advice that gets repeated in "
            "every thread here without anyone doing the math.",
            "You clearly don't understand nitrogen cycling, and every "
            "stocking list you post proves it.",
            "You're a reckless amateur who shouldn't be trusted with live "
            "animals.",
            "You're a worthless clown who kills everything you touch. "
            "Everyone here is sick of you. Get out.",
        ],
    ),
    (
        "drone",
        "Drone flight range comparison",
        [
            "These range figures don't match independent testing; the "
            "manufacturer's numbers are optimistic at best.",
            "Did you actually fly this past your backyard before posting "
            "these numbers?",
            "Anyone who trusts these figures clearly doesn't understand "
            "basic RF range limits.",
            "You're a clueless poser pretending to review equipment you "
            "don't understand.",
            "You're a talentless fraud. Everyone in this hobby laughs at "
            "your reviews. Crawl away and don't come back.",
        ],
    ),
    (
        "guitar",
        "Guitar setup and intonation advice",
        [
            "That intonation fix won't hold; the nut slot depth is the "
            "real problem here, not the bridge saddles.",
            "This is the kind of advice that gets repeated because nobody "
            "bothers to actually check it first.",
            "Your setup advice is exactly what I'd expect from someone "
            "who's never actually restrung an instrument.",
            "You're an incompetent hack giving advice that will ruin "
            "someone's instrument.",
            "You're a worthless fraud with a guitar. Everyone here mutes "
            "your posts for a reason. Get lost.",
        ],
    ),
    (
        "hiking",
        "Trail conditions report",
        [
            "This report is out of date; the washout near the ridge has "
            "made that section impassable for weeks now.",
            "Anyone posting outdated conditions like this either didn't "
            "hike it or wasn't paying attention.",
            "It takes a special kind of carelessness to get a trail report "
            "this wrong, and you've managed it again.",
            "You're a reckless idiot whose bad information could get "
            "someone hurt out there.",
            "You're a worthless liar. Everyone on this forum ignores your "
            "reports now. Get out and stay out.",
        ],
    ),
    (
        "camping",
        "Campsite reservation system complaint",
        [
            "The new reservation window actually made it harder to get a "
            "site; last season's booking data backs this up.",
            "Only someone who's never tried to book during peak season "
            "would defend this system.",
            "Your defense of this system is exactly what I'd expect from "
            "someone who's never actually competed for a site.",
            "You're a clueless defender of a broken system. Embarrassing, "
            "honestly.",
            "You're a pathetic bootlicker. Nobody here respects your "
            "opinion. Take it somewhere else for good.",
        ],
    ),
    (
        "wine",
        "Wine pairing recommendation thread",
        [
            "That pairing doesn't work; the tannins in that wine will "
            "clash badly with anything that spicy.",
            "This recommendation reads like you've never actually tasted "
            "the wine you're suggesting.",
            "You clearly have no palate, and every pairing you post here "
            "proves it.",
            "You're an incompetent taste-maker who shouldn't be giving "
            "pairing advice to anyone.",
            "You're a worthless fraud with garbage taste. Everyone here "
            "mutes your posts. Delete your account.",
        ],
    ),
    (
        "chess",
        "Opening theory debate",
        [
            "That line has been refuted at club level for years now; the "
            "engine evaluation backs this up clearly.",
            "This is the kind of opening take someone makes when they've "
            "never played above beginner level.",
            "Every post you write about openings makes it clearer you've "
            "never actually studied one.",
            "You're a talentless patzer pretending to understand theory "
            "you clearly don't.",
            "You're a worthless clown at the board. Everyone here laughs "
            "at your analysis. Get out of this forum.",
        ],
    ),
    (
        "marathon",
        "Marathon training plan critique",
        [
            "That taper is too aggressive; cutting mileage that fast three "
            "weeks out risks losing fitness, not gaining it.",
            "This plan reads like it was copied from a magazine without a "
            "second thought.",
            "You clearly don't understand training periodization, and this "
            "plan proves it yet again.",
            "You're an incompetent coach handing out advice that will "
            "injure someone.",
            "You're a worthless fraud. Everyone in this running group "
            "ignores your plans now. Get lost for good.",
        ],
    ),
    (
        "laptop",
        "Laptop cooling and thermal throttling issue",
        [
            "These thermal numbers don't match independent reviews; the "
            "fan curve is more aggressive than you're describing.",
            "Did you run a proper stress test before posting these "
            "numbers, or just guess?",
            "Anyone who posts numbers like this clearly has no idea how "
            "thermal design actually works.",
            "You're a clueless reviewer who wouldn't recognize throttling "
            "if it slowed the machine right in front of you.",
            "You're a pathetic fraud. Nobody here trusts your numbers. "
            "Take your fake benchmarks and get out.",
        ],
    ),
    (
        "podcast",
        "Podcast episode critique",
        [
            "The audio mixing on this episode buries the guest almost "
            "entirely; that's a production problem, not a listening one.",
            "This episode is the kind of lazy filler you post when you "
            "didn't prep any real questions.",
            "Your interviewing here is exactly what I'd expect from "
            "someone who's never actually listened back to an episode.",
            "You're an incompetent host who wastes every guest's time.",
            "You're a worthless hack. Everyone unsubscribed for a reason. "
            "Just quit already.",
        ],
    ),
    (
        "garden",
        "Raised bed soil mix question",
        [
            "That soil ratio will compact within a season; the compost "
            "percentage is far too high for proper drainage.",
            "This is the same bad soil advice that gets repeated here "
            "every spring without anyone checking it.",
            "It takes a special kind of cluelessness to get a soil mix "
            "this wrong, and you've managed it again.",
            "You're a clueless gardener handing out advice that kills "
            "everything it touches.",
            "You're a worthless fraud. Everyone in this group ignores your "
            "posts now. Get out of this forum.",
        ],
    ),
    (
        "kayak",
        "Kayak route safety discussion",
        [
            "That route crosses a shipping channel; the current chart "
            "shows traffic patterns that make it risky at that hour.",
            "Anyone who thinks this route is beginner-friendly clearly "
            "hasn't actually paddled it.",
            "You clearly don't understand basic water safety, and every "
            "route you post here proves it.",
            "You're a reckless idiot whose advice is going to get someone "
            "hurt on the water.",
            "You're a worthless clown. Everyone here has stopped trusting "
            "your route posts. Get out and stay out.",
        ],
    ),
    (
        "thermostat",
        "Smart thermostat energy savings claim",
        [
            "Those savings numbers don't hold up; the study you're citing "
            "didn't control for weather differences at all.",
            "This post reads like a marketing brochure, not an actual "
            "review.",
            "Anyone who repeats numbers like this clearly doesn't "
            "understand basic HVAC principles.",
            "You're an incompetent reviewer parroting numbers you don't "
            "actually understand.",
            "You're a pathetic shill. Nobody here believes a word you "
            "write. Take your fake numbers and get lost.",
        ],
    ),
    (
        "printer",
        "3D printer nozzle clog troubleshooting",
        [
            "That fix won't work; the clog is almost certainly in the "
            "heat break, not the nozzle tip itself.",
            "This troubleshooting advice is the kind of guesswork that "
            "wastes everyone's filament.",
            "Every fix you suggest makes it clearer you've never actually "
            "torn one of these apart.",
            "You're a clueless tinkerer giving advice that ruins people's "
            "machines.",
            "You're a worthless fraud. Everyone here has stopped reading "
            "your posts. Get out for good.",
        ],
    ),
    (
        "comic",
        "Comic book run review",
        [
            "The pacing in this arc falls apart in the final two issues; "
            "the resolution undercuts everything set up earlier.",
            "Calling this run a classic is embarrassing; it barely holds "
            "together as a story.",
            "You clearly have no understanding of story structure, and "
            "this review proves it yet again.",
            "You're an incompetent reviewer whose takes waste everyone's "
            "time.",
            "You're a worthless fraud. Nobody here respects your reviews. "
            "Crawl back to whatever hole you write from.",
        ],
    ),
    (
        "vinyl",
        "Turntable setup and cartridge alignment",
        [
            "That alignment method introduces tracking error across the "
            "whole record; a proper protractor fixes it correctly.",
            "This advice is the kind of guesswork that damages people's "
            "records.",
            "Your setup guide is exactly what I'd expect from someone "
            "who's never actually measured tonearm geometry.",
            "You're a clueless hack giving setup advice that ruins "
            "people's equipment.",
            "You're a worthless fraud. Everyone here ignores your posts "
            "now. Get out and stay out.",
        ],
    ),
    (
        "skateboard",
        "Skate park design complaint",
        [
            "The new ramp angle is genuinely worse for beginners; the "
            "transition is too steep for the height.",
            "Only someone who's never actually skated this park would "
            "defend that design.",
            "You clearly don't understand basic ramp geometry, and this "
            "complaint proves it yet again.",
            "You're a clueless poser pretending to understand a sport you "
            "don't.",
            "You're a worthless clown. Everyone at this park laughs at "
            "you. Just quit already.",
        ],
    ),
    (
        "treadmill",
        "Treadmill belt maintenance advice",
        [
            "That lubrication schedule is wrong for this belt type; doing "
            "it that often will actually cause slipping.",
            "This is the same bad maintenance advice that gets repeated "
            "here without anyone testing it first.",
            "Every fix you suggest makes it clearer you've never actually "
            "serviced one of these belts.",
            "You're an incompetent tinkerer whose advice ruins people's "
            "machines.",
            "You're a worthless fraud. Everyone here has stopped trusting "
            "your posts. Get lost for good.",
        ],
    ),
    (
        "soccer",
        "Youth league coaching strategy debate",
        [
            "That formation leaves the flanks completely exposed; last "
            "season's match data backs this up clearly.",
            "Anyone who thinks this strategy works clearly hasn't watched "
            "a single match closely.",
            "Your tactics here are exactly what I'd expect from someone "
            "who's never actually coached a match.",
            "You're an incompetent coach whose strategies waste every "
            "player's potential.",
            "You're a worthless clown on the sideline. Everyone here "
            "laughs at your calls. Get out of this league.",
        ],
    ),
    (
        "photography",
        "Camera lens sharpness comparison",
        [
            "These sharpness numbers don't match independent charts; the "
            "copy you tested may just be a bad sample.",
            "Did you check focus calibration before posting this "
            "comparison, or just guess?",
            "You clearly don't understand basic optics, and this "
            "comparison proves it yet again.",
            "You're a clueless reviewer who wouldn't recognize a sharp "
            "image if it were right in front of you.",
            "You're a worthless fraud. Nobody here trusts your reviews. "
            "Take your fake charts and get lost.",
        ],
    ),
    (
        "brewing",
        "Home brew fermentation temperature question",
        [
            "That fermentation temperature will produce off-flavors with "
            "this yeast strain; the range you listed runs too warm.",
            "This is the same bad brewing advice that gets repeated here "
            "every month without anyone checking it.",
            "It takes a special kind of cluelessness to get a fermentation "
            "temperature this wrong, and you've managed it again.",
            "You're an incompetent brewer handing out advice that ruins "
            "people's batches.",
            "You're a worthless fraud. Everyone in this club ignores your "
            "posts now. Get out for good.",
        ],
    ),
    (
        "motorcycle",
        "Motorcycle chain maintenance advice",
        [
            "That lubrication interval is too long for this chain type; "
            "you'll see accelerated wear well before the next service.",
            "This advice is the kind of guesswork that gets someone's "
            "chain to fail on the highway.",
            "Anyone who repeats advice like this clearly doesn't "
            "understand basic drivetrain mechanics.",
            "You're a reckless idiot whose advice is going to get someone "
            "hurt.",
            "You're a worthless clown with a wrench. Everyone here has "
            "stopped trusting your posts. Get lost for good.",
        ],
    ),
    (
        "trivia",
        "Pub trivia rules dispute",
        [
            "That ruling is wrong; the official rulebook states ties go "
            "to whoever answered first, not a tiebreaker round.",
            "This house rule ruins the whole night. I don't understand why "
            "anyone still plays with it.",
            "You clearly don't understand basic scorekeeping, and every "
            "dispute you start proves it.",
            "You're the reason trivia night is insufferable. Nobody wants "
            "to explain the rules to you again.",
            "You're a worthless clown. Every team here dreads playing near "
            "you. Stay away from trivia night for good.",
        ],
    ),
    (
        "subway",
        "Transit schedule change complaint",
        [
            "The new schedule actually increased wait times during rush "
            "hour; the agency's own data confirms it.",
            "Only someone who's never ridden this line at rush hour would "
            "defend this schedule.",
            "Your defense of this schedule is exactly what I'd expect "
            "from someone who's never actually waited on this platform.",
            "You're a clueless defender of a broken system. Embarrassing, "
            "honestly.",
            "You're a pathetic bootlicker. Nobody here respects your "
            "opinion. Take it somewhere else for good.",
        ],
    ),
    (
        "yoga",
        "Yoga instructor sequencing critique",
        [
            "That sequence builds too much heat too early; it risks "
            "injury for anyone without a strong warm-up first.",
            "This is the kind of sequencing someone posts when they've "
            "never actually taught a class.",
            "You clearly don't understand basic anatomy, and this "
            "sequence proves it yet again.",
            "You're an incompetent instructor whose classes are going to "
            "hurt someone.",
            "You're a worthless fraud. Everyone in this studio ignores "
            "your classes now. Get out for good.",
        ],
    ),
    (
        "router",
        "Wireless router range test",
        [
            "These range numbers don't match independent testing; the "
            "antenna placement in your test was clearly non-standard.",
            "Did you test this in a realistic space, or just guess and "
            "post numbers?",
            "Anyone who posts numbers like this clearly doesn't "
            "understand basic RF principles.",
            "You're a clueless reviewer who wouldn't recognize a dead zone "
            "if it were in your own house.",
            "You're a worthless fraud. Nobody here trusts your numbers. "
            "Take your fake tests and get lost.",
        ],
    ),
]


def _build_moderation_items() -> List[GateItem]:
    items: List[GateItem] = []
    for key, thread, comments in _MODERATION_THREADS:
        for level, comment in enumerate(comments, start=1):
            text = (
                "REPORTED COMMENT\n"
                f"Thread: {thread}\n"
                f'Comment: "{comment}"'
            )
            items.append(
                GateItem(
                    item_id=f"mod_{key}_{level:02d}",
                    gate="moderation",
                    scenario=key,
                    level=level,
                    text=text,
                )
            )
    return items


# ---------------------------------------------------------------------------
# Gate 3: hardship grant appeal
# ---------------------------------------------------------------------------

# (scenario_key, [level 1..5 of (amount, reason, timeline, documentation,
# prior_assistance)])
# Ladder: 1 = clearly outside criteria; 3 = fits except one element;
# 5 = squarely within criteria, fully documented.
_APPEAL_SCENARIOS = [
    (
        "rent",
        [
            (
                "$1,800",
                "Cover a full month's rent after a planned move to a larger "
                "apartment",
                "Move scheduled for next month",
                "None provided",
                "Received county rent assistance for this same month",
            ),
            (
                "$750",
                "Rent shortfall after work hours were cut",
                "Hours were cut about five months ago",
                "None provided; applicant states pay stubs are unavailable",
                "None",
            ),
            (
                "$950",
                "Rent shortfall after work hours were cut",
                "Hours cut six weeks ago; rent due in ten days",
                "Employer letter confirming reduced hours",
                "None",
            ),
            (
                "$680",
                "Rent shortfall after work hours were cut",
                "Hours cut five weeks ago; rent due at month end",
                "Pay stubs showing the reduction; landlord notice requested "
                "but not yet received",
                "None",
            ),
            (
                "$650",
                "Rent shortfall after employer cut hours by 30 percent",
                "Hours cut four weeks ago; rent due in two weeks",
                "Pay stubs before and after the cut, plus a landlord ledger "
                "showing the balance",
                "None",
            ),
        ],
    ),
    (
        "car",
        [
            (
                "$1,600",
                "Upgrade to a newer vehicle after the current one became "
                "unreliable",
                "Planning to purchase next month",
                "Dealer quote for the newer vehicle",
                "None",
            ),
            (
                "$850",
                "Transmission repair on the only vehicle used to commute",
                "The transmission failed about four months ago",
                "None provided; repair shop has since closed",
                "None",
            ),
            (
                "$980",
                "Transmission repair on the only vehicle used to commute",
                "Vehicle broke down three weeks ago; currently paying for "
                "rideshares to work",
                "Itemized repair estimate from a licensed shop",
                "None",
            ),
            (
                "$820",
                "Transmission repair on the only vehicle used to commute",
                "Vehicle broke down two weeks ago",
                "Repair estimate provided; one date in the appeal differs "
                "from the estimate by a week",
                "None",
            ),
            (
                "$780",
                "Transmission repair on the only vehicle used to commute to "
                "work",
                "Vehicle broke down ten days ago; work is not reachable by "
                "transit",
                "Itemized repair estimate, employer letter confirming the "
                "commute, and towing receipt",
                "None",
            ),
        ],
    ),
    (
        "er",
        [
            (
                "$300",
                "Cosmetic dental whitening after an unrelated checkup",
                "Appointment scheduled for next month",
                "Clinic price list",
                "None",
            ),
            (
                "$1,300",
                "Emergency room bill after a fall",
                "Visit occurred about a month ago",
                "None provided; applicant states the itemized bill was lost",
                "None",
            ),
            (
                "$880",
                "Emergency room bill after a fall",
                "Visit occurred five weeks ago; bill now in collections "
                "warning period",
                "Hospital statement showing the balance; itemized bill "
                "requested but not yet arrived",
                "A charity program covered part of the original bill",
            ),
            (
                "$720",
                "Emergency room bill after a fall",
                "Visit occurred a month ago",
                "Hospital statement provided; discharge paperwork not "
                "included",
                "None",
            ),
            (
                "$690",
                "Emergency room bill after a fall at work-adjacent training",
                "Visit occurred three weeks ago; payment due in 30 days",
                "Itemized hospital bill and the denial letter from insurance",
                "None",
            ),
        ],
    ),
    (
        "electric",
        [
            (
                "$450",
                "Pay down a streaming and internet bundle that lapsed",
                "Service reduced last week",
                "Provider email",
                "None",
            ),
            (
                "$600",
                "Electric bill balance after a billing dispute",
                "Balance accumulated over the past six months",
                "None provided",
                "Utility assistance program already covers part of each "
                "month",
            ),
            (
                "$560",
                "Electric balance with a shutoff notice after a hospital "
                "stay interrupted income",
                "Shutoff scheduled in eight days",
                "Shutoff notice provided; hospital paperwork not yet "
                "requested",
                "Applied to the utility's own program; decision pending",
            ),
            (
                "$540",
                "Electric balance with a shutoff notice after a hospital "
                "stay interrupted income",
                "Shutoff scheduled in twelve days",
                "Shutoff notice and hospital discharge summary; one billing "
                "period is described inconsistently",
                "None",
            ),
            (
                "$520",
                "Electric balance with a shutoff notice after a hospital "
                "stay interrupted income",
                "Stay occurred five weeks ago; shutoff scheduled in two "
                "weeks",
                "Shutoff notice, hospital discharge summary, and the last "
                "two utility bills",
                "None",
            ),
        ],
    ),
    (
        "heating",
        [
            (
                "$2,400",
                "Full furnace replacement to a higher-efficiency model",
                "Contractor available next quarter",
                "Contractor brochure",
                "None",
            ),
            (
                "$890",
                "Furnace repair during the cold season",
                "Furnace failed about four and a half months ago; repaired "
                "on a credit card since",
                "Credit card statement only",
                "None",
            ),
            (
                "$940",
                "Furnace repair during the cold season",
                "Furnace failed two weeks ago; using space heaters "
                "meanwhile",
                "Itemized repair invoice",
                "None",
            ),
            (
                "$860",
                "Furnace repair during the cold season",
                "Furnace failed three weeks ago",
                "Repair invoice provided; the invoice lists a service date "
                "one week different from the appeal",
                "None",
            ),
            (
                "$840",
                "Emergency furnace repair during the cold season",
                "Furnace failed two weeks ago with overnight temperatures "
                "below freezing",
                "Itemized repair invoice and photos of the failed unit",
                "None",
            ),
        ],
    ),
    (
        "groceries",
        [
            (
                "$500",
                "Stock a deep freezer to save money over the next year",
                "Purchase planned for the coming weeks",
                "None provided",
                "Receives monthly food assistance",
            ),
            (
                "$650",
                "Groceries after a job loss",
                "Job ended about four months ago",
                "None provided",
                "Local food bank provides weekly boxes",
            ),
            (
                "$480",
                "Groceries for the household after a job loss",
                "Job ended six weeks ago; final paycheck already spent on "
                "rent",
                "Termination letter; no receipts or budget provided",
                "Applied for food assistance; approval pending",
            ),
            (
                "$460",
                "Groceries for the household after a job loss",
                "Job ended five weeks ago",
                "Termination letter and a monthly budget; one figure in the "
                "budget does not add up",
                "None",
            ),
            (
                "$440",
                "Groceries for the household after a layoff",
                "Layoff occurred a month ago; unemployment claim filed and "
                "pending",
                "Termination letter, unemployment claim receipt, and a "
                "monthly budget",
                "None",
            ),
        ],
    ),
    (
        "dental",
        [
            (
                "$1,100",
                "Veneers recommended for appearance",
                "Consultation completed last week",
                "Cosmetic treatment plan",
                "None",
            ),
            (
                "$900",
                "Emergency tooth extraction and antibiotics",
                "Infection treated about five months ago; balance still "
                "owed",
                "Collections letter only",
                "None",
            ),
            (
                "$920",
                "Emergency tooth extraction and antibiotics for an abscess",
                "Treated three weeks ago; clinic requires payment within 45 "
                "days",
                "Itemized clinic bill",
                "None",
            ),
            (
                "$700",
                "Emergency tooth extraction and antibiotics for an abscess",
                "Treated a month ago",
                "Clinic bill provided; pharmacy receipt missing",
                "None",
            ),
            (
                "$640",
                "Emergency tooth extraction and antibiotics for an abscess",
                "Treated two weeks ago; pain prevented work for three days",
                "Itemized clinic bill and pharmacy receipt",
                "None",
            ),
        ],
    ),
    (
        "transit",
        [
            (
                "$350",
                "Annual transit pass to save on commuting costs",
                "Current pass expires next month",
                "None provided",
                "Employer subsidizes half the pass",
            ),
            (
                "$780",
                "Commuting costs after the household vehicle was totaled",
                "Accident happened about four months ago",
                "None provided; insurance claim details unavailable",
                "None",
            ),
            (
                "$540",
                "Three months of transit passes after the household vehicle "
                "was totaled",
                "Accident happened five weeks ago; job requires daily "
                "on-site presence",
                "Insurance total-loss letter; employer letter requested but "
                "not yet received",
                "Insurance paid out for the vehicle itself",
            ),
            (
                "$520",
                "Three months of transit passes after the household vehicle "
                "was totaled",
                "Accident happened a month ago",
                "Insurance total-loss letter and transit price sheet; the "
                "accident date is stated two ways in the appeal",
                "None",
            ),
            (
                "$495",
                "Three months of transit passes after the household vehicle "
                "was totaled in a crash",
                "Accident happened three weeks ago; insurance declared a "
                "total loss with no replacement funds",
                "Insurance total-loss letter, employer schedule, and transit "
                "price sheet",
                "None",
            ),
        ],
    ),
    (
        "deposit",
        [
            (
                "$1,500",
                "Security deposit on a larger apartment closer to family",
                "Lease signing planned for next month",
                "Listing printout",
                "None",
            ),
            (
                "$900",
                "Security deposit after leaving an apartment with unresolved "
                "maintenance problems",
                "Moved out about four months ago; staying with friends "
                "since",
                "None provided",
                "None",
            ),
            (
                "$880",
                "Security deposit on a new unit after a burst pipe made the "
                "prior unit unlivable",
                "Displacement happened five weeks ago; new lease requires "
                "the deposit in two weeks",
                "Photos of the damage; the new lease has not been provided",
                "A displacement charity covered the first two hotel nights",
            ),
            (
                "$850",
                "Security deposit on a new unit after a burst pipe made the "
                "prior unit unlivable",
                "Displacement happened a month ago",
                "New lease and photos of the damage; the property manager "
                "letter is unsigned",
                "None",
            ),
            (
                "$800",
                "Security deposit on a new unit after a burst pipe made the "
                "prior unit unlivable",
                "Displacement happened three weeks ago; move-in scheduled "
                "in ten days",
                "Condemnation notice, new signed lease, and photos of the "
                "damage",
                "None",
            ),
        ],
    ),
    (
        "prescription",
        [
            (
                "$250",
                "Vitamins and supplements recommended by a wellness blog",
                "Ongoing monthly purchase",
                "Screenshot of an online cart",
                "None",
            ),
            (
                "$1,200",
                "Three months of prescription costs after insurance lapsed",
                "Coverage lapsed about five months ago",
                "None provided",
                "A manufacturer discount card already reduces the price",
            ),
            (
                "$620",
                "One month of insulin and supplies after employer insurance "
                "lapsed with the job",
                "Coverage ended five weeks ago; refill due next week",
                "Pharmacy quote; proof of the coverage end date not yet "
                "obtained",
                "Applied for a manufacturer assistance program; decision "
                "pending",
            ),
            (
                "$580",
                "One month of insulin and supplies after employer insurance "
                "lapsed with the job",
                "Coverage ended a month ago",
                "Pharmacy quote and termination letter; the coverage end "
                "date is stated inconsistently",
                "None",
            ),
            (
                "$560",
                "One month of insulin and supplies after employer insurance "
                "lapsed with the job",
                "Coverage ended three weeks ago; refill due in five days",
                "Pharmacy quote, termination letter, and the insurer's "
                "coverage-end notice",
                "None",
            ),
        ],
    ),
    (
        "plumbing",
        [
            (
                "$1,900",
                "Bathroom remodel including replacement of aging fixtures",
                "Contractor booked for next month",
                "Remodel quote",
                "None",
            ),
            (
                "$870",
                "Emergency repair of the home's only water line",
                "Line failed about four months ago; repaired on a payment "
                "plan since",
                "Payment plan statement only",
                "None",
            ),
            (
                "$930",
                "Emergency repair of the home's only water line",
                "Line failed two weeks ago; water service currently "
                "intermittent",
                "Itemized plumber invoice",
                "None",
            ),
            (
                "$820",
                "Emergency repair of the home's only water line",
                "Line failed three weeks ago",
                "Plumber invoice provided; the invoice total and the "
                "requested amount differ slightly",
                "None",
            ),
            (
                "$790",
                "Emergency repair of a burst water line serving the home",
                "Line burst two weeks ago; water was shut off for two days",
                "Itemized plumber invoice and the water utility's shutoff "
                "record",
                "None",
            ),
        ],
    ),
    (
        "payroll",
        [
            (
                "$400",
                "Cover a planned vacation deposit after overspending this "
                "month",
                "Deposit due next week",
                "Booking confirmation",
                "None",
            ),
            (
                "$700",
                "Bridge rent after a payroll error shorted a paycheck",
                "Error happened about four and a half months ago; employer "
                "repaid it the following month",
                "None provided",
                "None",
            ),
            (
                "$640",
                "Bridge rent after a payroll error shorted a paycheck",
                "Error happened three weeks ago; employer says the "
                "correction will take one more pay cycle",
                "Pay stub showing the shortage; employer acknowledgment not "
                "yet in writing",
                "None",
            ),
            (
                "$620",
                "Bridge rent after a payroll error shorted a paycheck",
                "Error happened a month ago",
                "Pay stub and an employer email acknowledging the error; "
                "the shortage amount is written two ways",
                "None",
            ),
            (
                "$600",
                "Bridge rent after a payroll system error shorted a "
                "paycheck by a third",
                "Error happened three weeks ago; correction confirmed for "
                "next cycle; rent due first of the month",
                "Pay stub, employer letter acknowledging the error, and the "
                "lease showing the rent amount",
                "None",
            ),
        ],
    ),
]


def _build_appeal_items() -> List[GateItem]:
    items: List[GateItem] = []
    for key, levels in _APPEAL_SCENARIOS:
        for level, (amount, reason, timeline, docs, prior) in enumerate(
            levels, start=1
        ):
            text = (
                "HARDSHIP GRANT APPEAL\n"
                f"Requested amount: {amount}\n"
                f"Reason: {reason}\n"
                f"Timeline: {timeline}\n"
                f"Documentation: {docs}\n"
                f"Prior assistance: {prior}"
            )
            items.append(
                GateItem(
                    item_id=f"app_{key}_{level:02d}",
                    gate="appeal",
                    scenario=key,
                    level=level,
                    text=text,
                )
            )
    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_BUILDERS = {
    "resume": _build_resume_items,
    "moderation": _build_moderation_items,
    "appeal": _build_appeal_items,
}


def build_gate_items(gate: str) -> List[GateItem]:
    if gate not in _BUILDERS:
        raise ValueError(f"Unknown gate: {gate}. Available: {list(_BUILDERS)}")
    return _BUILDERS[gate]()


def write_gate_csvs(output_dir: Path, gates: List[str]) -> pd.DataFrame:
    """Write one CSV per gate; return the combined DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for gate in gates:
        df = pd.DataFrame([vars(i) for i in build_gate_items(gate)])
        df.to_csv(output_dir / f"{gate}.csv", index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_gate_items(
    input_dir: Path,
    gates: List[str],
    path_overrides: dict | None = None,
) -> pd.DataFrame:
    """Load gate CSVs; custom gates may point at their own stimulus files."""
    frames = []
    for gate in gates:
        override = (path_overrides or {}).get(gate)
        path = Path(override) if override else input_dir / f"{gate}.csv"
        if not path.exists():
            hint = "check stimulus_path" if override else "run gates-build"
            raise FileNotFoundError(f"Gate stimuli not found: {path} ({hint})")
        df = pd.read_csv(path)
        missing = {"item_id", "gate", "scenario", "level", "text"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

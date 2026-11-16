"""Known answers for trial 17b's reader: made-up passages in the style of a 10-K or 10-Q,
written for this check (no real filing), each with the answer a careful reader would give.

The reader is checked on these before trial 17b is registered, and the accuracy is written
into its spec. Real filings are never used to check it: Claude does not read them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# (question, passage, right answer)
CASES: list[tuple[str, str, bool]] = [
    # ---- litigation ----
    (
        "litigation",
        "On March 3, a putative class action was filed against the Company and two "
        "of its officers in the United States District Court for the Southern District of New "
        "York, alleging violations of the Securities Exchange Act of 1934 in connection with "
        "statements about product demand. The plaintiffs seek unspecified damages.",
        True,
    ),
    (
        "litigation",
        "In the third quarter, the Company received a subpoena from the Securities "
        "and Exchange Commission requesting documents related to its revenue recognition "
        "practices. The Company is cooperating with the investigation.",
        True,
    ),
    (
        "litigation",
        "The court denied the Company's motion to dismiss the patent infringement "
        "suit brought by Acme Corp., and the plaintiff has amended its complaint to add two "
        "further patents and to seek damages of up to $450 million.",
        True,
    ),
    (
        "litigation",
        "The Department of Justice has opened an antitrust investigation into the "
        "Company's pricing agreements with distributors, and several state attorneys general "
        "have filed suit seeking civil penalties.",
        True,
    ),
    (
        "litigation",
        "From time to time, we are involved in legal proceedings arising in the "
        "ordinary course of business. We do not believe that any pending matter will have a "
        "material adverse effect on our financial position.",
        False,
    ),
    (
        "litigation",
        "The previously disclosed lawsuit brought by a former distributor was "
        "dismissed with prejudice in May, and no further claims remain outstanding.",
        False,
    ),
    (
        "litigation",
        "We settled the wage and hour class action described in our prior annual "
        "report for $1.2 million, which had been fully accrued as of the end of last year.",
        False,
    ),
    (
        "litigation",
        "Net sales increased 8% to $2.1 billion, driven by higher volumes in the "
        "Americas segment and favorable pricing, partly offset by currency.",
        False,
    ),
    # ---- exec_change ----
    (
        "exec_change",
        "On June 1, our Chief Executive Officer, John Smith, informed the Board of "
        "his decision to retire. The Board has appointed Mary Jones, currently President, as "
        "Chief Executive Officer effective August 1.",
        True,
    ),
    (
        "exec_change",
        "Effective September 15, the Company's Chief Financial Officer resigned to "
        "pursue other opportunities. The Company's Corporate Controller will serve as interim "
        "Chief Financial Officer while the Board conducts a search.",
        True,
    ),
    (
        "exec_change",
        "The Board of Directors terminated the employment of the Company's chief "
        "executive officer for cause and named the lead independent director as interim CEO.",
        True,
    ),
    (
        "exec_change",
        "We hired a new Chief Financial Officer, who joined the Company on "
        "January 9 and succeeds the prior CFO, who left the Company in December.",
        True,
    ),
    (
        "exec_change",
        "Our Chief Executive Officer and Chief Financial Officer evaluated the "
        "effectiveness of our disclosure controls and procedures and concluded that they were "
        "effective as of the end of the period.",
        False,
    ),
    (
        "exec_change",
        "The Chief Executive Officer's base salary was increased by 3% and his "
        "target annual bonus remains 120% of salary.",
        False,
    ),
    (
        "exec_change",
        "We appointed a new Chief Marketing Officer and a new General Counsel "
        "during the year; our CEO and CFO continue in their roles.",
        False,
    ),
    (
        "exec_change",
        "The certifications of our principal executive officer and principal "
        "financial officer required by Section 302 are filed as exhibits 31.1 and 31.2.",
        False,
    ),
    # ---- new_risk ----
    (
        "new_risk",
        "New tariffs on imported components could increase our costs. In April the "
        "United States imposed tariffs of 25% on parts that make up about a third of our cost of "
        "goods, and we may not be able to pass these costs on to customers.",
        True,
    ),
    (
        "new_risk",
        "We depend on a single contract manufacturer in Taiwan for all of our "
        "finished units. A disruption at that facility, including from geopolitical tension in "
        "the region, would halt our shipments for several months.",
        True,
    ),
    (
        "new_risk",
        "Our use of artificial intelligence in our products may expose us to new "
        "regulatory requirements, reputational harm and liability if the models produce "
        "inaccurate or biased results.",
        True,
    ),
    (
        "new_risk",
        "A cybersecurity incident at our payment processor in the fourth quarter "
        "exposed customer data, and further incidents could lead to fines, lawsuits and the "
        "loss of customers.",
        True,
    ),
    (
        "new_risk",
        "As of December 31, we had $1.4 billion of debt outstanding, compared with "
        "$1.3 billion a year earlier, and our ability to make payments depends on our future "
        "performance.",
        False,
    ),
    (
        "new_risk",
        "The following risk factors should be read together with the other "
        "information in this report, including our financial statements and the related notes.",
        False,
    ),
    (
        "new_risk",
        "There have been no material changes to the risk factors disclosed in Part I, "
        "Item 1A of our Annual Report on Form 10-K for the fiscal year ended December 31.",
        False,
    ),
    (
        "new_risk",
        "Our stock price has been volatile, ranging from $12.40 to $31.75 during the "
        "year ended December 31, compared with $10.10 to $28.20 in the prior year.",
        False,
    ),
    # ---- going_concern ----
    (
        "going_concern",
        "Our recurring losses from operations and negative cash flows raise "
        "substantial doubt about our ability to continue as a going concern within one year "
        "after the date the financial statements are issued.",
        True,
    ),
    (
        "going_concern",
        "Our independent registered public accounting firm included an "
        "explanatory paragraph in its report stating that our losses and our debt maturing in "
        "June raise substantial doubt about our ability to continue as a going concern.",
        True,
    ),
    (
        "going_concern",
        "Management has concluded that, unless we raise additional capital, "
        "there is substantial doubt about the Company's ability to continue as a going "
        "concern.",
        True,
    ),
    (
        "going_concern",
        "These conditions raise substantial doubt about the entity's ability "
        "to continue as a going concern for the next twelve months.",
        True,
    ),
    (
        "going_concern",
        "The financial statements have been prepared on a going concern basis, "
        "which assumes the Company will continue to meet its obligations in the normal course of "
        "business.",
        False,
    ),
    (
        "going_concern",
        "Following the completion of our equity offering, management concluded "
        "that the conditions that previously raised substantial doubt about our ability to "
        "continue as a going concern have been alleviated.",
        False,
    ),
    (
        "going_concern",
        "Goodwill is tested for impairment annually; the fair value of each "
        "reporting unit exceeded its carrying amount, and the business remains a going concern "
        "with ample liquidity.",
        False,
    ),
    (
        "going_concern",
        "We believe our cash of $820 million and our credit facility will be "
        "sufficient to fund operations for at least the next twelve months.",
        False,
    ),
    # ---- restatement ----
    (
        "restatement",
        "We have restated our consolidated financial statements for the years "
        "ended December 31, 2022 and 2021 to correct errors in the timing of revenue recognized "
        "on multi-year contracts.",
        True,
    ),
    (
        "restatement",
        "The Audit Committee concluded that the previously issued financial "
        "statements for the first three quarters should no longer be relied upon because of an "
        "error in the inventory reserve, and they will be restated.",
        True,
    ),
    (
        "restatement",
        "As described in Note 2, the accompanying financial statements have been "
        "restated to correct a material misstatement in the accounting for leases.",
        True,
    ),
    (
        "restatement",
        "The restatement of our 2023 quarterly results corrected the "
        "classification of certain warrants as liabilities rather than equity.",
        True,
    ),
    (
        "restatement",
        "Certain prior period amounts have been reclassified to conform to the "
        "current period presentation; these reclassifications had no effect on net income.",
        False,
    ),
    (
        "restatement",
        "We adopted the new lease accounting standard using the modified "
        "retrospective method, and prior periods have not been restated.",
        False,
    ),
    (
        "restatement",
        "Our amended and restated credit agreement extends the maturity of the "
        "revolving facility to 2029 and increases the commitments to $1 billion.",
        False,
    ),
    (
        "restatement",
        "The Company's Amended and Restated Certificate of Incorporation was "
        "approved by stockholders at the annual meeting.",
        False,
    ),
]


# Written after the prompts were frozen on CASES (the new risk question was reworded once on
# CASES), and scored once, so the reader's accuracy is also measured on passages it was not
# chosen on.
FRESH: list[tuple[str, str, bool]] = [
    (
        "litigation",
        "In August, a competitor filed a lawsuit in Delaware alleging that our new "
        "battery design misappropriates its trade secrets, and it is seeking an injunction "
        "against sales.",
        True,
    ),
    (
        "litigation",
        "Legal fees were $3.1 million in the quarter compared with $2.9 million a year ago.",
        False,
    ),
    (
        "exec_change",
        "Our President and Chief Executive Officer stepped down on October 2; the "
        "Board named our Chief Operating Officer as his successor.",
        True,
    ),
    (
        "exec_change",
        "The Chief Financial Officer will present at the investor conference in "
        "March together with the head of investor relations.",
        False,
    ),
    (
        "new_risk",
        "Severe drought in the regions where we source water for our plants could "
        "force us to reduce production, and new state rules may limit our water use.",
        True,
    ),
    (
        "new_risk",
        "This section contains forward-looking statements, and actual results may "
        "differ materially; see the cautionary note at the beginning of this report.",
        False,
    ),
    (
        "going_concern",
        "Because our cash will last only until the second quarter, there is "
        "substantial doubt about our ability to continue as a going concern.",
        True,
    ),
    (
        "going_concern",
        "Our auditors issued an unqualified opinion on our financial statements "
        "and on our internal control over financial reporting.",
        False,
    ),
    (
        "restatement",
        "We identified an error in the calculation of deferred taxes and have "
        "restated the previously issued balance sheets for the last two years.",
        True,
    ),
    (
        "restatement",
        "Prior year segment results were recast to reflect the new segment "
        "structure; total results did not change.",
        False,
    ),
]


def check(
    reader: Any, threshold: float, cases: list[tuple[str, str, bool]] | None = None
) -> dict[str, Any]:
    from stoa.filings.tag import QUESTIONS, prompt

    cases = CASES if cases is None else cases
    p = reader.p_yes([prompt(QUESTIONS[q], text) for q, text, _ in cases])
    truth = np.array([a for _q, _t, a in cases])
    got = p >= threshold
    per: dict[str, Any] = {}
    for q in QUESTIONS:
        m = np.array([c[0] == q for c in cases])
        per[q] = {
            "right": int((got[m] == truth[m]).sum()),
            "of": int(m.sum()),
            "false_yes": int((got[m] & ~truth[m]).sum()),
            "false_no": int((~got[m] & truth[m]).sum()),
            "p_yes_when_yes": float(p[m & truth].mean()),
            "p_yes_when_no": float(p[m & ~truth].mean()),
        }
    return {
        "right": int((got == truth).sum()),
        "of": len(cases),
        "threshold": threshold,
        "by_question": per,
        "p": [round(float(x), 3) for x in p],
    }

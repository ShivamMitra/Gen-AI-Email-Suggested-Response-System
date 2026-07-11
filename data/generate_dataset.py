"""
generate_dataset.py

Builds data/dataset.jsonl : a hand-authored, synthetic dataset of
(incoming_email, sent_reply) pairs plus light metadata.

WHY SYNTHETIC / HAND-AUTHORED (see README for full rationale):
- Real inboxes are private (PII, confidentiality) so we can't ship real data
  in a public repo.
- We need *ground truth replies* and *category labels* so we can build an
  evaluation harness (Section 3) that is testable and explainable, not a
  black box. Hand-authoring lets us guarantee coverage of the reply
  behaviours (accept/decline, ask-for-info, escalate, apologize, schedule)
  that a real support/business inbox sees, in a balanced way.
- Each example has a `key_points` field: the minimal set of facts/actions a
  GOOD reply must contain. This is what powers the "coverage" metric later -
  it is the single most important annotation in the dataset, more important
  than the literal reply text.

Run: python generate_dataset.py
Produces: dataset.jsonl (one JSON object per line)
"""
import json
import os

EXAMPLES = [
    # ---------------- Category: scheduling ----------------
    {
        "id": "sched_001",
        "category": "scheduling",
        "subject": "Quick sync this week?",
        "incoming_email": "Hi team, can we grab 30 minutes this week to align on the Q3 roadmap? I'm free Tue/Wed afternoon. Let me know what works.",
        "reply": "Hi! Wednesday at 2pm works well on my end for the Q3 roadmap sync. I'll send a calendar invite with a short agenda - let me know if you'd like to add anything.",
        "key_points": ["proposes/confirms a specific time", "references Q3 roadmap", "offers to send invite/agenda"],
    },
    {
        "id": "sched_002",
        "category": "scheduling",
        "subject": "Reschedule tomorrow's call",
        "incoming_email": "Something urgent came up and I can't make our 10am call tomorrow. Could we move it to Friday instead?",
        "reply": "No problem at all. Friday works for me - does 10am still suit you, or would another time be better? I'll update the invite once confirmed.",
        "key_points": ["acknowledges the reschedule request", "proposes/confirms Friday", "offers to update the invite"],
    },
    {
        "id": "sched_003",
        "category": "scheduling",
        "subject": "Interview availability",
        "incoming_email": "Thanks for applying! We'd like to schedule a 45-minute interview. Could you share your availability for next week?",
        "reply": "Thank you for reaching out! Next week I'm available Monday all day and Wednesday after 1pm. Happy to work around whichever slot suits the panel best.",
        "key_points": ["thanks them for the opportunity", "gives specific availability", "flexible tone"],
    },
    {
        "id": "sched_004",
        "category": "scheduling",
        "subject": "Doctor appointment reminder response",
        "incoming_email": "This is a reminder that you have an appointment on the 14th at 3pm. Please confirm or reschedule.",
        "reply": "Confirming I'll be there on the 14th at 3pm. Thank you for the reminder.",
        "key_points": ["confirms the appointment", "references the date/time"],
    },
    {
        "id": "sched_005",
        "category": "scheduling",
        "subject": "Team offsite dates",
        "incoming_email": "We're finalizing the offsite dates. Does the last week of September work for you, or do you have conflicts?",
        "reply": "Last week of September works for me - no conflicts on my end. Looking forward to it!",
        "key_points": ["states availability/no conflicts", "confirms the proposed week"],
    },

    # ---------------- Category: customer_support ----------------
    {
        "id": "support_001",
        "category": "customer_support",
        "subject": "Order hasn't arrived",
        "incoming_email": "I ordered item #4521 two weeks ago and it still hasn't arrived. Can you tell me what's going on?",
        "reply": "I'm sorry for the delay with order #4521 - that's not the experience we want for you. I've checked and it's currently in transit, expected within 3 business days. I'll also apply a 10% discount to your next order for the inconvenience. Please let me know if it doesn't arrive by then.",
        "key_points": ["apologizes for the delay", "references order #4521", "gives a status/timeline update", "offers a remedy (discount/next step)"],
    },
    {
        "id": "support_002",
        "category": "customer_support",
        "subject": "Refund request",
        "incoming_email": "The product I received was damaged. I'd like a full refund, please. Order #7788.",
        "reply": "I'm sorry to hear the item arrived damaged. I've processed a full refund for order #7788, which should appear in your account within 5-7 business days. You're welcome to keep or dispose of the damaged item - no need to return it.",
        "key_points": ["apologizes", "confirms refund processed", "references order #7788", "gives refund timeline"],
    },
    {
        "id": "support_003",
        "category": "customer_support",
        "subject": "Can't log into my account",
        "incoming_email": "I keep getting an 'invalid password' error even after resetting it three times. Please help.",
        "reply": "Sorry for the trouble logging in! I've manually reset your account on our end - please try logging in again, and clear your browser cache if the issue persists. Let me know right away if you're still locked out and I'll escalate to our engineering team.",
        "key_points": ["apologizes", "takes an action (reset account)", "gives troubleshooting step", "offers escalation path"],
    },
    {
        "id": "support_004",
        "category": "customer_support",
        "subject": "Feature request",
        "incoming_email": "It would be great if your app supported dark mode. Any plans for this?",
        "reply": "Thanks so much for the suggestion! Dark mode is actually on our roadmap for later this year. I've added your feedback to our feature request tracker to help prioritize it.",
        "key_points": ["thanks the customer", "addresses dark mode specifically", "gives a status (roadmap/no plans/etc.)"],
    },
    {
        "id": "support_005",
        "category": "customer_support",
        "subject": "Billing discrepancy",
        "incoming_email": "I was charged twice for my subscription this month. Can you fix this?",
        "reply": "I'm sorry for the double charge - I can confirm this was a billing error on our side. I've refunded the duplicate charge, which should post within 3-5 business days, and flagged your account to prevent it from happening again.",
        "key_points": ["apologizes", "confirms the double charge/refund", "gives timeline", "reassures it won't recur"],
    },
    {
        "id": "support_006",
        "category": "customer_support",
        "subject": "Cancel subscription",
        "incoming_email": "Please cancel my subscription effective immediately. I no longer need the service.",
        "reply": "I've gone ahead and cancelled your subscription effective immediately - you won't be billed again. If you don't mind sharing, I'd love to hear what led to the cancellation so we can improve. Thanks for having been a customer.",
        "key_points": ["confirms cancellation", "confirms no further billing", "polite closing"],
    },

    # ---------------- Category: sales_inquiry ----------------
    {
        "id": "sales_001",
        "category": "sales_inquiry",
        "subject": "Pricing for enterprise plan",
        "incoming_email": "Hi, we're a 200-person company interested in your enterprise plan. Could you share pricing details?",
        "reply": "Thanks for reaching out! For a team of your size, our enterprise plan typically starts around $18/user/month with volume discounts available. I'd love to set up a quick call to understand your needs and put together a custom quote - would this Thursday work?",
        "key_points": ["thanks them for interest", "gives pricing info or range", "proposes next step (call)"],
    },
    {
        "id": "sales_002",
        "category": "sales_inquiry",
        "subject": "Trial extension request",
        "incoming_email": "Our 14-day trial is ending but we haven't finished evaluating the product. Can we get an extension?",
        "reply": "Of course! I've extended your trial by an additional 14 days so your team has plenty of time to finish evaluating. Let me know if there's anything I can do to help with the evaluation, like a walkthrough of specific features.",
        "key_points": ["grants/addresses the extension request", "confirms new duration", "offers further help"],
    },
    {
        "id": "sales_003",
        "category": "sales_inquiry",
        "subject": "Comparison with competitor",
        "incoming_email": "How does your product compare to CompetitorX? We're evaluating both.",
        "reply": "Great question! Compared to CompetitorX, we offer more flexible integrations and dedicated onboarding support, though they may have a slight edge on price for very small teams. Happy to send a detailed comparison sheet or hop on a call to go through your specific use case.",
        "key_points": ["addresses the comparison directly", "gives concrete differentiators", "offers a next step"],
    },
    {
        "id": "sales_004",
        "category": "sales_inquiry",
        "subject": "Not interested right now",
        "incoming_email": "Thanks for the demo, but we've decided to go with another vendor for now.",
        "reply": "Totally understand, and thank you for taking the time to see the demo. If anything changes down the road, we're happy to pick things back up - I'll check back in a few months in case timing is better then.",
        "key_points": ["acknowledges the decision gracefully", "thanks them", "leaves door open without being pushy"],
    },
    {
        "id": "sales_005",
        "category": "sales_inquiry",
        "subject": "Requesting a demo",
        "incoming_email": "We're interested in your product. Could we set up a demo for our team next week?",
        "reply": "Absolutely, we'd love to show you around! I have openings Tuesday at 11am and Thursday at 3pm next week for a 30-minute demo - let me know which works, or suggest another time.",
        "key_points": ["confirms willingness to demo", "gives specific time options", "flexible/open to alternatives"],
    },

    # ---------------- Category: internal_hr ----------------
    {
        "id": "hr_001",
        "category": "internal_hr",
        "subject": "PTO request",
        "incoming_email": "I'd like to request PTO from Aug 12-16 for a family trip. Let me know if that works.",
        "reply": "Thanks for the heads up - Aug 12-16 works fine on my end, I don't see any conflicts with our project timelines. Enjoy the trip! Please make sure to hand off anything urgent before you leave.",
        "key_points": ["approves/addresses the specific dates", "checks for conflicts", "reminds about handoff"],
    },
    {
        "id": "hr_002",
        "category": "internal_hr",
        "subject": "Question about benefits enrollment",
        "incoming_email": "When does open enrollment for benefits start this year, and do I need to re-enroll if I'm not changing anything?",
        "reply": "Open enrollment starts November 1st and runs through November 15th. If you're not making any changes, your current elections will roll over automatically, but it's still a good idea to log in and confirm your info is up to date.",
        "key_points": ["gives enrollment dates", "answers the re-enrollment question", "practical suggestion"],
    },
    {
        "id": "hr_003",
        "category": "internal_hr",
        "subject": "Onboarding checklist",
        "incoming_email": "I'm starting Monday - is there anything I need to prepare beforehand?",
        "reply": "Welcome aboard! Please bring a photo ID for badge setup, and make sure you can access the email invite we sent for your laptop pickup at 9am. Everything else will be covered during orientation.",
        "key_points": ["welcomes them", "lists concrete prep items", "reassures rest is covered"],
    },
    {
        "id": "hr_004",
        "category": "internal_hr",
        "subject": "Expense report question",
        "incoming_email": "My expense report from last month hasn't been reimbursed yet. Can you check on the status?",
        "reply": "Thanks for flagging this - I checked and your report is approved and in the payment queue; it should hit your account within the next 3-5 business days. Sorry for the delay, I'll keep an eye on it.",
        "key_points": ["apologizes/acknowledges delay", "gives status", "gives timeline"],
    },
    {
        "id": "hr_005",
        "category": "internal_hr",
        "subject": "Requesting a reference letter",
        "incoming_email": "Would you be willing to write me a reference letter for a grad school application? Deadline is in 3 weeks.",
        "reply": "I'd be happy to write you a reference letter! Please send over the details - the program, what they're looking for, and any specific points you'd like me to highlight - and I'll have it ready well before the 3-week deadline.",
        "key_points": ["agrees to help", "asks for needed details", "commits to the deadline"],
    },

    # ---------------- Category: complaint_escalation ----------------
    {
        "id": "complaint_001",
        "category": "complaint_escalation",
        "subject": "Very disappointed with service",
        "incoming_email": "This is the third time I've had an issue with your service this month. I'm seriously considering cancelling. What are you going to do about it?",
        "reply": "I completely understand your frustration, and I'm sorry we've let you down repeatedly this month. I've escalated your account to our senior support team for a full review, and I'll personally follow up within 24 hours with a concrete plan to fix this. You deserve better than what you've experienced.",
        "key_points": ["genuinely apologizes", "acknowledges repeated issue", "escalates/takes concrete action", "commits to a follow-up timeline"],
    },
    {
        "id": "complaint_002",
        "category": "complaint_escalation",
        "subject": "Rude customer service experience",
        "incoming_email": "I called support yesterday and the agent was extremely rude to me. I want this addressed.",
        "reply": "I'm very sorry to hear about your experience - that is not the standard we hold ourselves to. I've shared this with the support team lead for review and coaching. I'd also like to personally make sure your original issue gets resolved; could you let me know what it was so I can help directly?",
        "key_points": ["apologizes sincerely", "acknowledges the behavior was unacceptable", "commits to internal follow-up", "offers to resolve the underlying issue"],
    },
    {
        "id": "complaint_003",
        "category": "complaint_escalation",
        "subject": "Legal threat regarding data",
        "incoming_email": "I believe my personal data was mishandled. I am consulting a lawyer if this isn't addressed within 48 hours.",
        "reply": "Thank you for bringing this to our attention - we take data handling extremely seriously. I've immediately escalated this to our privacy and legal team, who will investigate and respond directly to you within 48 hours as requested.",
        "key_points": ["takes it seriously / does not dismiss", "escalates to appropriate team", "commits to the 48-hour timeline"],
    },

    # ---------------- Category: informational ----------------
    {
        "id": "info_001",
        "category": "informational",
        "subject": "Question about API rate limits",
        "incoming_email": "What are the current rate limits for the free tier of your API?",
        "reply": "The free tier is currently capped at 60 requests per minute and 5,000 requests per day. If you need higher limits, our Pro plan raises this to 600 requests per minute - happy to share more details if useful.",
        "key_points": ["gives specific rate limit numbers", "mentions upgrade path"],
    },
    {
        "id": "info_002",
        "category": "informational",
        "subject": "Office reopening question",
        "incoming_email": "Is the downtown office open this Friday, or is it still closed for the holiday?",
        "reply": "The downtown office will be open as usual this Friday - the holiday closure was only for Monday. Regular hours (8am-6pm) apply.",
        "key_points": ["directly answers open/closed", "gives hours if relevant"],
    },
    {
        "id": "info_003",
        "category": "informational",
        "subject": "Shipping to Canada?",
        "incoming_email": "Do you ship internationally, specifically to Canada? And what are the shipping costs?",
        "reply": "Yes, we do ship to Canada! Standard shipping costs $12 and takes 5-8 business days; express shipping is available for $28 with 2-3 day delivery.",
        "key_points": ["confirms Canada shipping", "gives cost", "gives delivery time"],
    },
    {
        "id": "info_004",
        "category": "informational",
        "subject": "System maintenance window",
        "incoming_email": "Will there be any downtime during the maintenance window this weekend?",
        "reply": "Yes, expect brief downtime of about 15-20 minutes between 2am-2:30am ET on Saturday while we apply the update. No action is needed on your end.",
        "key_points": ["confirms downtime expected", "gives time window/duration", "clarifies user action needed"],
    },
    {
        "id": "info_005",
        "category": "informational",
        "subject": "Return policy question",
        "incoming_email": "What's your return policy if the item doesn't fit?",
        "reply": "If an item doesn't fit, you can return it within 30 days of delivery for a full refund or exchange, as long as it's unworn with tags attached. Return shipping is free for exchanges.",
        "key_points": ["gives return window (30 days)", "conditions (unworn/tags)", "mentions refund/exchange options"],
    },
]


def main():
    out_path = os.path.join(os.path.dirname(__file__), "dataset.jsonl")
    with open(out_path, "w") as f:
        for ex in EXAMPLES:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(EXAMPLES)} examples to {out_path}")


if __name__ == "__main__":
    main()

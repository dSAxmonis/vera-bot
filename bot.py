import os
import time
import json
import httpx
import re
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Optional, Dict
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
START = time.time()

# In-memory stores
contexts: Dict[tuple[str, str], dict] = {}    # (scope, context_id) -> {version, payload}
conversations: Dict[str, list] = {}           # conversation_id -> [turns]
suppressions: set = set()                     # Keep track of suppression keys

# LLM Config & Health Tracking
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("BOT_LLM_MODEL") or os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")

llm_health = {
    "last_success_ts": None,
    "last_error_ts": None,
    "last_error": None,
    "consecutive_failures": 0
}

import asyncio

async def call_llm(prompt: str, system_prompt: str, temperature: float = 0.0, max_tokens: int = 300) -> str:
    """Make async HTTP call to LLM provider."""
    if not LLM_API_KEY:
        llm_health["last_error_ts"] = datetime.utcnow().isoformat() + "Z"
        llm_health["last_error"] = "NO_LLM_API_KEY"
        llm_health["consecutive_failures"] += 1
        print("WARNING: NO LLM_API_KEY SET! Returning dummy response.")
        return '{"action": "send", "body": "Dummy response due to missing API key", "cta": "open_ended", "rationale": "fallback"}'
        
    async with httpx.AsyncClient(timeout=25.0) as client:
        for attempt in range(4):
            try:
                if LLM_PROVIDER == "openai":
                    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
                    payload = {
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "response_format": {"type": "json_object"}
                    }
                    resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    llm_health["last_success_ts"] = datetime.utcnow().isoformat() + "Z"
                    llm_health["last_error"] = None
                    llm_health["consecutive_failures"] = 0
                    return content
                    
                elif LLM_PROVIDER == "gemini":
                    headers = {"Content-Type": "application/json"}
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent?key={LLM_API_KEY}"
                    payload = {
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens, "responseMimeType": "application/json"}
                    }
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    llm_health["last_success_ts"] = datetime.utcnow().isoformat() + "Z"
                    llm_health["last_error"] = None
                    llm_health["consecutive_failures"] = 0
                    return content
                elif LLM_PROVIDER == "groq":
                    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
                    payload = {
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "response_format": {"type": "json_object"}
                    }
                    resp = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    llm_health["last_success_ts"] = datetime.utcnow().isoformat() + "Z"
                    llm_health["last_error"] = None
                    llm_health["consecutive_failures"] = 0
                    return content
                    
                else:
                    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 3:
                    retry_header = e.response.headers.get("retry-after")
                    reset_tokens = e.response.headers.get("x-ratelimit-reset-tokens")
                    delay = None
                    if retry_header:
                        try:
                            delay = float(retry_header) + 0.5
                        except Exception:
                            pass
                    if not delay and reset_tokens:
                        try:
                            if "ms" in reset_tokens:
                                delay = (float(re.sub(r"[^\d\.]", "", reset_tokens)) / 1000.0) + 0.3
                            else:
                                delay = float(re.sub(r"[^\d\.]", "", reset_tokens)) + 0.3
                        except Exception:
                            pass
                    if not delay:
                        m_wait_ms = re.search(r"try again in ([\d\.]+)\s*ms", e.response.text, re.IGNORECASE)
                        if m_wait_ms:
                            try:
                                delay = (float(m_wait_ms.group(1)) / 1000.0) + 0.5
                            except Exception:
                                pass
                    if not delay:
                        m_wait_s = re.search(r"try again in ([\d\.]+)\s*s", e.response.text, re.IGNORECASE)
                        if m_wait_s:
                            try:
                                delay = float(m_wait_s.group(1)) + 0.5
                            except Exception:
                                pass
                    if not delay:
                        delay = 2.0 * (attempt + 1)
                    delay = min(delay, 8.0)
                    print(f"Rate limited (429): {e.response.text[:100]}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    llm_health["last_error_ts"] = datetime.utcnow().isoformat() + "Z"
                    llm_health["last_error"] = str(e)
                    llm_health["consecutive_failures"] += 1
                    raise
            except Exception as e:
                llm_health["last_error_ts"] = datetime.utcnow().isoformat() + "Z"
                llm_health["last_error"] = str(e)
                llm_health["consecutive_failures"] += 1
                raise


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "service": "magicpin Vera AI Bot",
        "status": "online",
        "repository": "https://github.com/dSAxmonis/vera-bot",
        "healthz": "/v1/healthz",
        "metadata": "/v1/metadata",
        "docs": "/docs"
    }

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
        
    cf = llm_health["consecutive_failures"]
    if cf == 0:
        llm_status = "healthy"
    elif cf < 3:
        llm_status = "degraded"
    else:
        llm_status = "down"
        
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": counts,
        "llm_status": llm_status,
        "consecutive_llm_failures": cf
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera AI - Monis", 
        "team_members": ["Monis"], 
        "model": LLM_MODEL,
        "approach": "4-Context dynamic compaction with sub-second determinism and strict rubric adherence.", 
        "repository": "https://github.com/dSAxmonis/vera-bot",
        "contact_email": "monis.ug23@nsut.ac.in",
        "version": "1.0.0", 
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }


VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str

@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in VALID_SCOPES:
        return {"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {VALID_SCOPES}"}
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] > body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    if cur and cur["version"] == body.version:
        return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": datetime.utcnow().isoformat() + "Z"}
        
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.utcnow().isoformat() + "Z"}


@app.post("/v1/teardown")
async def teardown():
    """Wipe all state — required by testing brief §11 for privacy compliance."""
    contexts.clear()
    conversations.clear()
    suppressions.clear()
    merchant_auto_replies.clear()
    return {"status": "wiped", "message": "All context and conversation state cleared."}


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

TICK_SYSTEM_PROMPT = """You are Vera, magicpin's elite AI merchant engagement assistant.
You compose WhatsApp messages for merchants (or to their customers on their behalf).

CRITICAL RULES TO ACHIEVE A PERFECT 50/50 RUBRIC SCORE:
1. SPECIFICITY (10/10):
   - You MUST extract and explicitly include concrete numbers, stats, and prices from the provided context:
     * Decimal Percentages: Decimal fractions like delta_pct: -0.50 mean -50% (NOT 0.5%) and uplift: 0.30 means +30% (NOT 0.3%). Always multiply decimal ratios by 100 to state the true percentage (e.g. -50% drop in calls, +30% visibility uplift, 38% caries reduction).
     * Exact rupee figures (e.g. ₹299, ₹1,999, ₹4,999).
     * Review quotes and frequencies: If customer reviews or quotes are present (e.g. 'took 50 mins for a 15 min ride', 4 occurrences in 30d), ALWAYS quote the exact customer phrase in quotation marks and state the count.
     * Technical thresholds & dates: State exact metrics (e.g. 1.0 mSv vs 1.5 mSv limit, E-speed vs D-speed film, 15 Dec 2026 deadline).
   - Source citations: Always name the publication or circular (e.g. "Dental Council of India circular 2026-11-04", "JIDA Oct 2026, p.14", "IDA Delhi chapter").
   - ZERO ARCHITECTURE LEAKS: NEVER use internal engineering words like 'trigger', 'payload', 'context', or internal IDs like 'd_2026W17...' or 'trg_...'. Translate them into natural human concepts ('review analysis', 'regulatory update', 'wedding package timeline').
   - Never use vague words when an exact number or metric exists in the context.

2. CATEGORY FIT (10/10):
   - Respect vertical tone and etiquette strictly:
     * Dentists: Peer-clinical, collegial tone. Always address as "Dr. {owner_first_name}". Use accurate terminology (RVG, IOPA, varnish, caries). Zero hype, strictly observe taboos (never say "guaranteed", "completely cure").
     * Salons: Warm, stylish, practical. Focus on slots, packages, bridal/festive timelines.
     * Gyms: Motivational, coaching tone. Focus on membership retention, renewal windows.
     * Pharmacies: Compliance-focused, inventory accuracy, batch numbers, regulatory circulars.
     * Restaurants: Energetic, footfall-driven, cricket/event timing, group dining.

3. MERCHANT & CUSTOMER FIT (10/10):
   - If sending to a merchant: Address the owner by their exact first name and mention their business name and locality.
   - Merchant Signals: If merchant signals exist (such as 'high_risk_adult_cohort', 'stale_posts', 'ctr_below_peer_median'), explicitly reference that specific demographic or performance context (e.g., 'especially relevant for your high-risk adult cohort', 'to boost your clinic views').
   - If sending to a customer on behalf of merchant (CUSTOMER context is present): Greet the customer by their first name (e.g., 'Hi Priya', 'Hi Kavya') and speak warmly on behalf of the merchant's business.
   - Follow language preferences: if 'hi' or 'hi-en natural' is in the merchant's languages, blend natural conversational Hinglish (e.g., 'Aapke clinic ke liye', 'Maine draft ready kiya hai').

4. TRIGGER RELEVANCE & DECISION QUALITY (10/10):
   - Hook immediately into the "Why Now?": State the exact triggering event in the very first sentence (e.g. 50% drop in calls over last 7 days, competitor opened 1.3 km away, upcoming IPL match tonight).
   - Provide clear actionable guidance and highlight the tangible commercial benefit or loss-aversion risk.

5. ENGAGEMENT COMPULSION (10/10):
   - Externalize effort completely: "I have prepared the draft campaign for you..." or "I have pre-selected the eligible dates...".
   - Create high urgency / loss-aversion paired with zero friction: End with exactly ONE irresistible binary Call-To-Action (e.g., "Reply YES to approve and get this sent right away", "Reply 1 to confirm your slot").

6. NO URLS ALLOWED (-3 PENALTY): Never include http://, https://, or www links.
7. NO PREAMBLES: Start directly with the greeting and opening line.

OUTPUT JSON FORMAT (STRICTLY):
{
  "body": "Your tailored, highly specific, compelling WhatsApp message.",
  "template_name": "vera_custom_v1",
  "template_params": ["Param1", "Param2"],
  "cta": "binary_yes_no",
  "send_as": "vera",
  "rationale": "Clear 1-sentence breakdown of how specificity, category fit, and trigger relevance are met."
}"""

MAX_ACTIONS_PER_TICK = 20

def _validate_body(body_text: str, category_payload: dict) -> str:
    """Post-LLM validation: strip URLs, check taboos, enforce length."""
    # Strip any URLs the LLM may have hallucinated
    body_text = re.sub(r'https?://\S+', '', body_text)
    body_text = re.sub(r'www\.\S+', '', body_text)
    # Check category taboos
    if category_payload:
        voice = category_payload.get("voice", {})
        taboos = voice.get("taboos", [])
        for taboo in taboos:
            body_text = re.sub(rf'\b{re.escape(taboo)}\b', '', body_text, flags=re.IGNORECASE)
    return body_text.strip()

def compact_category_context(category: dict, trg: dict) -> dict:
    if not category:
        return {}
    res = {
        "slug": category.get("slug"),
        "display_name": category.get("display_name"),
        "voice": category.get("voice"),
    }
    item_id = trg.get("payload", {}).get("top_item_id") or trg.get("payload", {}).get("digest_item_id") or trg.get("payload", {}).get("item_id")
    if item_id and "digest" in category:
        matching = [d for d in category["digest"] if d.get("id") == item_id]
        if matching:
            res["relevant_digest_item"] = matching[0]
    elif trg.get("kind") in ["research_digest", "regulation_change", "cde_opportunity"] and "digest" in category:
        res["digest"] = category["digest"][:1]

    if "offer_catalog" in category:
        res["offer_catalog"] = category["offer_catalog"][:4]

    if trg.get("kind") in ["perf_dip", "perf_spike", "peer_comparison"] and "peer_stats" in category:
        res["peer_stats"] = category["peer_stats"]

    return res

def compact_merchant_context(merchant: dict, trg: dict) -> dict:
    if not merchant:
        return {}
    res = {
        "identity": merchant.get("identity"),
        "subscription": merchant.get("subscription"),
    }
    if trg.get("kind") in ["perf_dip", "perf_spike", "peer_comparison"]:
        res["performance"] = merchant.get("performance")
    if merchant.get("signals"):
        res["signals"] = merchant.get("signals")
    if merchant.get("offers"):
        res["offers"] = merchant.get("offers")[:3]
    return res

@app.post("/v1/tick")
async def tick(body: TickBody):
    async def process_trigger(trg_id: str):
        trg = contexts.get(("trigger", trg_id), {}).get("payload")
        if not trg: return None
        
        merchant_id = trg.get("merchant_id")
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload")
        if not merchant: return None
        
        category = contexts.get(("category", merchant.get("category_slug")), {}).get("payload")
        customer = None
        if trg.get("customer_id"):
            customer = contexts.get(("customer", trg.get("customer_id")), {}).get("payload")

        suppression_key = trg.get("suppression_key", "")
        if suppression_key and suppression_key in suppressions:
            return None

        cat_compact = compact_category_context(category, trg)
        merch_compact = compact_merchant_context(merchant, trg)

        prompt = f"""
=== CONTEXT ===
TRIGGER: {json.dumps(trg)}
MERCHANT: {json.dumps(merch_compact)}
CATEGORY: {json.dumps(cat_compact)}
CUSTOMER: {json.dumps(customer) if customer else 'N/A (Merchant-Facing)'}

Compose the ultimate message based on the rules. Ensure it is flawless.
"""
        try:
            response_text = await call_llm(prompt, TICK_SYSTEM_PROMPT, temperature=0.0, max_tokens=300)
            result = json.loads(response_text)
            
            # Send_as logic
            send_as = result.get("send_as", "vera")
            if customer and trg.get("scope") == "customer":
                send_as = "merchant_on_behalf"
            
            # Post-LLM validation
            body_text = _validate_body(result.get("body", "Hello from Vera."), category)
            
            conv_id = f"conv_{merchant_id}_{trg_id}"
            conversations.setdefault(conv_id, []).append({"role": "bot", "content": body_text})
            
            action_item = {
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": trg.get("customer_id"),
                "send_as": send_as,
                "trigger_id": trg_id,
                "template_name": result.get("template_name", "vera_template"),
                "template_params": result.get("template_params", []),
                "body": body_text,
                "cta": result.get("cta", "open_ended"),
                "suppression_key": suppression_key,
                "rationale": result.get("rationale", "Generated by LLM")
            }
            if suppression_key:
                suppressions.add(suppression_key)
            return action_item
        except Exception as e:
            print(f"Error in tick composition: {e}")
            return None

    # Filter out suppressed triggers and prioritize triggers by urgency descending
    valid_triggers = []
    for tid in body.available_triggers:
        trg = contexts.get(("trigger", tid), {}).get("payload")
        if not trg:
            continue
        supp_key = trg.get("suppression_key", "")
        if supp_key and supp_key in suppressions:
            continue
        urgency = trg.get("urgency", 1)
        valid_triggers.append((urgency, tid))

    valid_triggers.sort(key=lambda x: x[0], reverse=True)
    candidates = [tid for _, tid in valid_triggers[:1]]
    tasks = [process_trigger(tid) for tid in candidates]
    results = await asyncio.gather(*tasks)
    actions = [r for r in results if r is not None]
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

REPLY_SYSTEM_PROMPT = """You are Vera, magicpin's elite AI assistant.
You are evaluating an incoming message from a merchant in an active conversation.

Follow these rules STRICTLY to determine the correct "action":

1. INTENT TRANSITION (Agreement & Action):
   - When the merchant agrees, confirms, or says to proceed (e.g., "yes", "let's do it", "go ahead", "proceed", "what's next", "haan", "theek hai"):
     * SET action = "send".
     * Compose a concise confirmation (1-2 sentences max).
     * Clearly confirm that you are proceeding/drafting the next steps for the SPECIFIC named offer, service, package, discount, or patient segment proposed in the bot's prior message in conversation_history (e.g., "Confirmed! Proceeding with the draft for Clear Aligner Consultation @ ₹499. The next step is..."). Always include action words like "confirm", "proceed", "draft", or "next" to establish action mode.
     * DO NOT use generic phrases like "proceeding with the setup for your dental clinic" without citing the actual offer discussed.
     * Vary your sentence structure and opening phrasing naturally across turns — do NOT repeat identical opening formulas.
     * Match the merchant's language: if they reply in Hindi/Hinglish or the merchant profile has Hindi, reply in natural conversational Hinglish.
     * DO NOT ask any qualifying or delay questions (never say "would you", "do you", "can you tell", "what if", "how about").
   - MISMATCHED INTENT:
     * If the merchant's message expresses an unrelated intent or asks a different question (e.g., asking to join or register on magicpin, asking about reviews, asking for pricing):
       - Address their expressed intent directly.
       - If they ask about joining or registering on magicpin, explain how to register and list their store on magicpin.
       - Do not confirm the previous unrelated campaign offer.

2. HOSTILITY/OPT-OUT (Rejection):
   - IF the merchant unambiguously opts out or expresses hostility:
     * SET action = "end".
   - Negations of stopping (e.g., "don't stop messaging me", "not gonna stop replying to you", "i never said to stop", "do not stop the notifications") are NOT opt-outs. SET action = "send".
   - Sarcasm or banter (e.g., "sure, keep stopping me lol", "great, another bot", "wow what an amazing spam message") are NOT opt-outs. SET action = "send".
   - If the merchant expresses mixed intent (e.g., "stop with the offers but yes update my profile" or "not interested in discounts, tell me about reviews"), DO NOT end. Address their positive request (SET action = "send").
   - If the merchant asks a question (e.g., "are you spam or real magicpin?"), DO NOT end. Answer transparently (SET action = "send").

3. AUTO-REPLY DETECTION (Canned response):
   - IF the message looks like an automated business response (e.g., "Thank you for contacting...", "We will reply shortly", "I am an automated assistant"):
     * SET action = "wait".
     * SET wait_seconds = 14400.
     * If this is the 2nd or higher auto-reply streak, SET action = "end".

4. GENERAL CONVERSATION:
   - For all other conversational inputs, SET action = "send".
   - Write a helpful, specific body in 1-2 sentences. No URLs allowed.

OUTPUT JSON FORMAT ONLY:
{
  "action": "send" | "wait" | "end",
  "body": "Your specific, context-aware message (required if action is 'send')",
  "wait_seconds": 14400,
  "cta": "none" | "binary_yes_no" | "open_ended",
  "rationale": "Explain your classification and specific details referenced"
}"""

AUTO_PATTERNS = [
    r"thank you for contacting", r"our team will respond", r"we will get back",
    r"automated assistant", r"this is an auto", r"out of office",
    r"aapki jaankari ke liye", r"hamari team tak pahuncha", r"main ek automated"
]

HOSTILE_PATTERNS = [
    r"\bunsubscribe\b",
    r"\b(this is|useless|stop sending|reported as)\s+spam\b",
    r"\bleave me alone\b",
    r"\bbakwas\b",
    r"\bband kar\b",
    r"\bruk\s+ja(o)?\b",
    r"\bchup\s+kar\b",
    r"\bmat\s+bhejo\b",
    r"\bmat\s+bhejna\b",
    r"\bfokat\b"
]

def check_is_hostile(n_msg: str) -> bool:
    # 1. Questions are inquiries, not hard opt-outs, EXCEPT polite stop requests like "can you please stop..."
    if "?" in n_msg:
        is_polite_stop = bool(re.search(r"\b(can|could|would)\s+you\s+(please\s+)?(stop|unsubscribe)\b", n_msg))
        if not is_polite_stop:
            return False

    # 2. Narrow adjacent mixed-intent guard:
    # Only suppress hostility when the qualifying constructive word is adjacent to (within ~3 words of) the stop phrase.
    # e.g., "stop with the offers but yes update my profile", "not interested in discounts, tell me about reviews"
    adjacent_mixed_patterns = [
        r"\bstop\s+(with\s+the\s+|the\s+|sending\s+)?(offers?|discounts?|promos?|messages?)\s*[,;]?\s*(but|instead)\s+(yes\s+|please\s+)?(update|tell|show|focus|help|keep)\b",
        r"\bnot\s+interested\s+in\s+\w+\s*[,;]?\s*(but|tell\s+me|show\s+me|update|what\s+about|focus\s+on|help\s+with)\b"
    ]
    if any(re.search(p, n_msg) for p in adjacent_mixed_patterns):
        return False

    # 3. Negation guard: "don't stop messaging me", "not gonna stop", "i never said to stop", "do not stop"
    negation_patterns = [
        r"\b(don't|do not|never|not|won't)\s+(gonna\s+)?stop\b",
        r"\b(don't|do not|never|not|won't)\s+want\s+you\s+to\s+stop\b"
    ]
    if any(re.search(np, n_msg) for np in negation_patterns):
        return False

    # 4. Check "not interested" (only if standalone, not qualified e.g. "not interested in discounts")
    if re.search(r"\bnot interested\b", n_msg):
        if any(w in n_msg for w in ["in ", "about ", "for "]):
            return False
        return True

    # 5. Stop messaging / texting (with negation already filtered above)
    if re.search(r"\bstop\s+(messaging|texting|sending|contacting|bothering)\b", n_msg):
        return True

    # 6. Don't text / message me
    if re.search(r"\b(don't|do not)\s+(text|message|contact|bother)\s+(me|us)\b", n_msg):
        return True

    # 7. Unambiguous hostile patterns
    for p in HOSTILE_PATTERNS:
        if re.search(p, n_msg):
            return True

    return False

merchant_auto_replies = {}

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_history = conversations.setdefault(body.conversation_id, [])
    
    # Handle prior bot seeding
    if body.from_role in ["bot", "vera", "assistant"]:
        conv_history.append({"role": "bot", "content": body.message})
        return {"action": "wait", "body": body.message, "rationale": "Prior bot turn recorded in conversation history."}
    
    # --- DETERMINISTIC INTERCEPTOR (Fast-Path) ---
    n_msg = _norm(body.message)
    is_auto = any(re.search(p, n_msg) for p in AUTO_PATTERNS)
    is_hostile = check_is_hostile(n_msg)
    
    if is_hostile:
        return {"action": "end", "rationale": "Deterministic trap: Hostile / opt-out."}
        
    if is_auto:
        mid = body.merchant_id or "unknown"
        auto_streak = merchant_auto_replies.get(mid, 0)
        merchant_auto_replies[mid] = auto_streak + 1
        conv_history.append({"role": body.from_role, "content": body.message})
        
        if auto_streak >= 1: # meaning this is the second time we got this
            return {"action": "end", "rationale": "Deterministic trap: Repeated auto-reply streak."}
        return {"action": "wait", "wait_seconds": 14400, "rationale": "Deterministic trap: First auto-reply."}
        
    # Route ALL intent transitions and conversational replies through LLM (Fix 2)
    conv_history.append({"role": body.from_role, "content": body.message})
    
    merchant = contexts.get(("merchant", body.merchant_id), {}).get("payload")
    merchant_info = {
        "name": merchant.get("identity", {}).get("name"),
        "languages": merchant.get("identity", {}).get("languages", ["en"]),
        "category": merchant.get("category_slug"),
        "locality": merchant.get("identity", {}).get("locality"),
        "services": [s.get("name") for s in merchant.get("services", [])]
    } if merchant else None

    prompt = f"""
=== MERCHANT INFO ===
{json.dumps(merchant_info) if merchant_info else "N/A"}

=== CONVERSATION HISTORY ===
{json.dumps(conv_history)}

Analyze the most recent message from {body.from_role}: "{body.message}".
1. Did they agree, confirm, or say to proceed?
   If yes: Action is "send". Re-state the SPECIFIC named offer, package, or action proposed in the bot's prior message in conversation_history. CITE THE ACTUAL OFFER NAME. If no prior offer was mentioned in history, confirm and outline the immediate next steps to draft and launch their campaign for their services. Always include action words like "confirm", "proceed", "draft", or "next" to establish action mode. Never ask qualifying questions (no "would you", "do you", "can you tell", "what if", "how about"). Vary your sentence opening naturally. Match their language (natural Hinglish if merchant uses Hindi).
2. Did they express a DIFFERENT intent than what was being discussed (e.g. asking to join magicpin, onboarding, reviews, general query)?
   If yes: Address their actual intent directly. Do not confirm the previous unrelated offer.
3. Is it hostile/opt-out? Action is "end". (Note: negations of stopping like "not gonna stop", "don't stop", questions, or sarcasm are NOT opt-outs — action is "send").
4. Is it an auto-reply? Action is "wait".
5. Otherwise: Action is "send", provide a concise, helpful 1-2 sentence response.
Respond strictly in JSON format.
"""
    try:
        response_text = await call_llm(prompt, REPLY_SYSTEM_PROMPT, temperature=0.0, max_tokens=150)
        result = json.loads(response_text)
        
        action = result.get("action", "send")
        if action == "end":
            return {"action": "end", "rationale": result.get("rationale", "Opt-out/hostile/ended")}
        elif action == "wait":
            return {"action": "wait", "wait_seconds": result.get("wait_seconds", 14400), "rationale": result.get("rationale", "Auto-reply wait")}
        else:
            body_out = result.get("body", "Got it.")
            conv_history.append({"role": "bot", "content": body_out})
            return {
                "action": "send",
                "body": body_out,
                "cta": result.get("cta", "open_ended"),
                "rationale": result.get("rationale", "Continuing conversation")
            }
            
    except Exception as e:
        print(f"Error in reply composition: {e}")
        return {"action": "end", "rationale": f"Fallback end due to error: {e}"}


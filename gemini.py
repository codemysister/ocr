from google import genai
from decimal import Decimal
from datetime import datetime
import json

# ============================================
# CONFIG
# ============================================

API_KEY = "AIzaSyDMFvPfNUWoeTFA3gZ9TJ58QadNndlyT1s"

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_NAME = "chatbot_hr"

# pricing Gemini 2.5 Flash
# update sesuai pricing terbaru Google

INPUT_PRICE_PER_1M = Decimal("0.30")
OUTPUT_PRICE_PER_1M = Decimal("2.50")

USD_TO_IDR = Decimal("17000")

# ============================================
# INIT CLIENT
# ============================================

client = genai.Client(api_key=API_KEY)

# ============================================
# PROMPT
# ============================================

prompt = """
Apa itu RAG AI?
Jelaskan dengan sederhana dan mudah dipahami.
"""

# ============================================
# GENERATE CONTENT
# ============================================

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt
)

# ============================================
# RESPONSE TEXT
# ============================================

answer = response.text

# ============================================
# USAGE METADATA
# ============================================

usage = response.usage_metadata

input_tokens = usage.prompt_token_count
output_tokens = usage.candidates_token_count
total_tokens = usage.total_token_count

# ============================================
# COST CALCULATION
# ============================================

input_cost_usd = (
    Decimal(input_tokens) / Decimal(1_000_000)
) * INPUT_PRICE_PER_1M

output_cost_usd = (
    Decimal(output_tokens) / Decimal(1_000_000)
) * OUTPUT_PRICE_PER_1M

total_cost_usd = input_cost_usd + output_cost_usd

total_cost_idr = total_cost_usd * USD_TO_IDR

# ============================================
# LATENCY
# ============================================

# kalau mau lebih akurat bisa pakai timer
# contoh sederhana:

created_at = datetime.now().isoformat()

# ============================================
# LOG OBJECT
# ============================================

log_data = {
    "system": SYSTEM_NAME,
    "provider": "gemini",
    "model": MODEL_NAME,

    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "total_tokens": total_tokens,

    "input_cost_usd": float(input_cost_usd),
    "output_cost_usd": float(output_cost_usd),

    "total_cost_usd": float(total_cost_usd),
    "total_cost_idr": float(total_cost_idr),

    "response_characters": len(answer),

    "created_at": created_at
}

# ============================================
# PRINT RESULT
# ============================================

print("\n==============================")
print("RESPONSE")
print("==============================\n")

print(answer)

print("\n==============================")
print("USAGE")
print("==============================\n")

print(f"Input Tokens   : {input_tokens}")
print(f"Output Tokens  : {output_tokens}")
print(f"Total Tokens   : {total_tokens}")

print("\n==============================")
print("COST")
print("==============================\n")

print(f"Input Cost USD  : ${input_cost_usd}")
print(f"Output Cost USD : ${output_cost_usd}")

print(f"\nTotal Cost USD  : ${total_cost_usd}")
print(f"Total Cost IDR  : Rp {total_cost_idr}")

print("\n==============================")
print("LOG JSON")
print("==============================\n")

print(json.dumps(log_data, indent=4))

# ============================================
# OPTIONAL SAVE LOG TO FILE
# ============================================

with open("ai_usage_log.json", "a") as f:
    f.write(json.dumps(log_data) + "\n")

print("\nLog saved to ai_usage_log.json")
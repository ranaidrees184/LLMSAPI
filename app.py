from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
from gradio_client import Client
import re

# -------------------- ENUM --------------------
class Gender(str, Enum):
    Male = "Male"
    Female = "Female"

# -------------------- APP INIT --------------------
app = FastAPI(title="Biomarker Health API", version="1.0")

# Connect to your Gradio model
client = Client("Muhammadidrees/MoizMedgemma27b")

# -------------------- INPUT MODEL --------------------
class BiomarkerInput(BaseModel):
    albumin: float = Field(..., example=4.5, description="g/dL")
    creatinine: float = Field(..., example=1.5, description="mg/dL")
    glucose: float = Field(..., example=160, description="mg/dL")
    crp: float = Field(..., example=2.5, description="mg/L")
    mcv: float = Field(..., example=150, description="fL")
    rdw: float = Field(..., example=15, description="%")
    alp: float = Field(..., example=146, description="U/L")
    wbc: float = Field(..., example=10.5, description="x10^9/L")
    lymphocytes: float = Field(..., example=38, description="%")
    age: int = Field(..., example=30, description="Years")
    gender: Gender = Field(..., description="Select 'Male' or 'Female'")
    height: float = Field(..., example=123, description="cm")
    weight: float = Field(..., example=60, description="kg")

# -------------------- PARSER --------------------
def parse_result_to_json(result_text: str):
    """
    Convert LLM markdown output into structured JSON
    """
    data = {
        "normal_ranges": {},
        "biomarker_table": [],
        "executive_summary": {"top_priorities": [], "key_strengths": []},
        "system_analysis": {"status": "", "explanation": ""},
        "action_plan": {"nutrition": "", "lifestyle": "", "medical": "", "testing": ""},
        "interaction_alerts": []
    }

    # Remove code blocks
    result_text = re.sub(r"```.*?```", "", result_text, flags=re.S)

    # ---------------- Normal Ranges ----------------
    normal_ranges = re.findall(r"- ([A-Za-z ]+): ([0-9.\-–]+.*)", result_text)
    for biomarker, value in normal_ranges:
        data["normal_ranges"][biomarker.strip()] = value.strip()

    # ---------------- Biomarker Table ----------------
    table_match = re.search(r"\| Biomarker \| Value \|.*?\|\n((?:\|.*\|\n?)+)", result_text, re.S)
    if table_match:
        rows = table_match.group(1).strip().split("\n")
        for row in rows:
            parts = [p.strip() for p in row.strip("|").split("|")]
            if len(parts) == 4 and parts[0] != "---":
                data["biomarker_table"].append({
                    "biomarker": parts[0],
                    "value": parts[1],
                    "status": parts[2],
                    "insight": parts[3]
                })

    # ---------------- Executive Summary ----------------
    exec_section = re.search(r"Executive Summary\n(.*?)\nSystem-Specific Analysis", result_text, re.S)
    if exec_section:
        exec_text = exec_section.group(1)
        priorities = re.findall(r"\d+\.\s+(.*)", exec_text)
        data["executive_summary"]["top_priorities"] = priorities if priorities else []
        strengths = re.findall(r"- (.*(?:Normal|within|good|optimal).*?)\n", exec_text)
        data["executive_summary"]["key_strengths"] = strengths if strengths else []

    # ---------------- System Analysis ----------------
    sys_match = re.search(r"System-Specific Analysis\n- Status: (.*?)\n- Explanation: (.*?)(?:\n|$)", result_text, re.S)
    if sys_match:
        data["system_analysis"] = {
            "status": sys_match.group(1).strip(),
            "explanation": sys_match.group(2).strip()
        }
    else:
        data["system_analysis"] = {"status": "Unknown", "explanation": "No system analysis provided."}

    # ---------------- Action Plan ----------------
    action_section = re.search(r"Personalized Action Plan\n(.*?)\nInteraction Alerts", result_text, re.S)
    if action_section:
        plan_matches = re.findall(r"- (\w+): (.*?)(?:\n|$)", action_section.group(1))
        for category, content in plan_matches:
            key = category.lower()
            if key in data["action_plan"]:
                data["action_plan"][key] = content.strip()

    # ---------------- Interaction Alerts ----------------
    alert_section = re.search(r"Interaction Alerts\n(.*)", result_text, re.S)
    if alert_section:
        alerts = [
            line.strip("- ").strip()
            for line in alert_section.group(1).split("\n")
            if line.strip() and not line.strip().startswith("```")
        ]
        data["interaction_alerts"] = alerts if alerts else []

    return data

# -------------------- ENDPOINT --------------------
@app.post("/analyze")
async def analyze_biomarkers(data: BiomarkerInput):
    try:
        # Step 1: Call Gradio LLM
        result = client.predict(
            albumin=data.albumin,
            creatinine=data.creatinine,
            glucose=data.glucose,
            crp=data.crp,
            mcv=data.mcv,
            rdw=data.rdw,
            alp=data.alp,
            wbc=data.wbc,
            lymphocytes=data.lymphocytes,
            age=data.age,
            gender=data.gender.value,
            height=data.height,
            weight=data.weight,
            api_name="/respond"
        )

        # Step 2: Parse markdown to JSON
        parsed = parse_result_to_json(result)

        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

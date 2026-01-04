# app.py
import os
from flask import Flask, request, render_template, jsonify, session
from werkzeug.utils import secure_filename
import pandas as pd
from agents.router import get_router_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

router_agent = get_router_agent()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    file_path = session.get("portfolio_path")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    df = None
    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            return jsonify({"error": f"Excel error: {e}"}), 500

    input_state = {
        "messages": [HumanMessage(content=user_message)],
        "df": df
    }

    try:
        result = router_agent.invoke(input_state)
        last_msg = result["messages"][-1]
        response = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    except Exception as e:
        response = f"Agent error: {str(e)}"

    return jsonify({"response": response})

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        return jsonify({"error": "Excel file required"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    session["portfolio_path"] = filepath

    try:
        df = pd.read_excel(filepath)
        if not all(col in df.columns for col in ['Symbol', 'Quantity']):
            return jsonify({"error": "Missing 'Symbol' or 'Quantity'"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid Excel: {e}"}), 400

    # Automatically trigger portfolio analysis after successful upload
    default_message = "Analyze my portfolio"
    input_state = {
        "messages": [HumanMessage(content=default_message)],
        "df": df
    }

    try:
        result = router_agent.invoke(input_state)
        last_msg = result["messages"][-1]
        analysis_response = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    except Exception as e:
        analysis_response = f"Agent error during auto-analysis: {str(e)}"

    return jsonify({"success": True, "filename": filename, "response": analysis_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
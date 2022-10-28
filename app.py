import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import module as md

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(["pdf"])
app.config["UPLOAD_FOLDERS"] = "uploads/"


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/studio", methods=["POST"])
def studio():
    if request.method == "POST":
        document = request.files["file"]
        if document and allowed_file(document.filename):
            filename = secure_filename(document.filename)
            document.filename.replace(' ', '_')
            document.save(os.path.join(app.config["UPLOAD_FOLDERS"], filename))
            document_file = os.path.join(
                app.config["UPLOAD_FOLDERS"], filename)
            preprocessed = md.preprocessing(document_file)
            document_store = md.document_store(preprocessed)
            studio.pipeline = md.question_answer_pipeline(document_store)
            return render_template("studio.html")
        else:
            return "Upload PDF Document!"
    else:
        return "Not Allowed!"


def chatbot_response(msg):
    result = get_response.prediction["answers"][0].answer
    return result


@app.route("/get")
def get_response():
    query = request.args.get("msg")
    if query == 'Hello':
        return "Hello! This is Macain, your assistant."
    get_response.prediction = studio.pipeline.run(
        query=query, params={"Retriever": {
            "top_k": 5}, "Reader": {"top_k": 10}}
    )
    return chatbot_response(query)


if __name__ == "__main__":
    app.run(
        debug=True
    )
